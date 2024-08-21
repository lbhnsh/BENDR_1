import re
from sys import gettrace

# from dn3.trainable.utils import _make_mask, _make_span_from_seeds
# from dn3.data.dataset import DN3ataset
# from dn3.utils import LabelSmoothedCrossEntropyLoss
# from dn3.trainable.models import Classifier
# from dn3.transforms.batch import BatchTransform

# Swap these two for Ipython/Jupyter
# import tqdm
# import tqdm.notebook as tqdm
import tqdm.auto as tqdm

import torch
# ugh the worst, why did they make this protected...
from torch.optim.lr_scheduler import _LRScheduler as Scheduler

import numpy as np
from pandas import DataFrame
from collections import OrderedDict
from torch.utils.data import DataLoader, WeightedRandomSampler

import copy
from copy import deepcopy
from torch import nn

from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import ConcatDataset, DataLoader

import yaml
import mne
import mne.io as loader
from fnmatch import fnmatch
from pathlib import Path

from parse import search
import moabb.datasets as mbd
from mne import pick_types, read_annotations, set_log_level
import warnings
from mne.io.constants import FIFF
from abc import ABC

from mne.utils._bunch import NamedInt
from torch.nn.functional import interpolate
from collections.abc import Iterable
import bisect
from torch.utils.data.dataset import random_split

def same_channel_sets(channel_sets: list):
    """Validate that all the channel sets are consistent, return false if not"""
    for chs in channel_sets[1:]:
        if chs.shape[0] != channel_sets[0].shape[0] or chs.shape[1] != channel_sets[0].shape[1]:
            return False
        # if not np.all(channel_sets[0] == chs):
        #     return False
    return True


class InstanceTransform(object):

    def __init__(self, only_trial_data=True):
        """
        Trial transforms are, for the most part, simply operations that are performed on the loaded tensors when they are
        fetched via the :meth:`__call__` method. Ideally this is implemented with pytorch operations for ease of execution
        graph integration.
        """
        self.only_trial_data = only_trial_data

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, *x):
        """
        Modifies a batch of tensors.
        Parameters
        ----------
        x : torch.Tensor, tuple
            The trial tensor, not including a batch-dimension. If initialized with `only_trial_data=False`, then this
            is a tuple of all ids, labels, etc. being propagated.
        Returns
        -------
        x : torch.Tensor, tuple
            The modified trial tensor, or tensors if not `only_trial_data`
        """
        raise NotImplementedError()

    def new_channels(self, old_channels):
        """
        This is an optional method that indicates the transformation modifies the representation and/or presence of
        channels.

        Parameters
        ----------
        old_channels : ndarray
                       An array whose last two dimensions are channel names and channel types.

        Returns
        -------
        new_channels : ndarray
                      An array with the channel names and types after this transformation. Supports the addition of
                      dimensions e.g. a list of channels into a rectangular grid, but the *final two dimensions* must
                      remain the channel names, and types respectively.
        """
        return old_channels

    def new_sfreq(self, old_sfreq):
        """
        This is an optional method that indicates the transformation modifies the sampling frequency of the underlying
        time-series.

        Parameters
        ----------
        old_sfreq : float

        Returns
        -------
        new_sfreq : float
        """
        return old_sfreq

    def new_sequence_length(self, old_sequence_length):
        """
        This is an optional method that indicates the transformation modifies the length of the acquired extracts,
        specified in number of samples.

        Parameters
        ----------
        old_sequence_length : int

        Returns
        -------
        new_sequence_length : int
        """
        return old_sequence_length

class Preprocessor:
    """
    Base class for various preprocessing actions. Sub-classes are called with a subclass of `_Recording`
    and operate on these instances in-place.

    Any modifications to data specifically should be implemented through a subclass of :any:`BaseTransform`, and
    returned by the method :meth:`get_transform()`
    """
    def __call__(self, recording, **kwargs):
        """
        Preprocess a particular recording. This is allowed to modify aspects of the recording in-place, but is not
        strictly advised.

        Parameters
        ----------
        recording :
        kwargs : dict
                 New :any:`_Recording` subclasses may need to provide additional arguments. This is here for support of
                 this.
        """
        raise NotImplementedError()

    def get_transform(self):
        """
        Generate and return any transform associated with this preprocessor. Should be used after applying this
        to a dataset, i.e. through :meth:`DN3ataset.preprocess`

        Returns
        -------
        transform : BaseTransform
        """
        raise NotImplementedError()

class DN3atasetNanFound(BaseException):
    """
    Exception to be triggered when DN3-dataset variants load NaN data, or data becomes NaN when pushed through
    transforms.
    """
    pass


class DN3ataset(TorchDataset):

    def __init__(self):
        """
        Base class for that specifies the interface for DN3 datasets.
        """
        self._transforms = list()
        self._safe_mode = False
        self._mutli_proc_start = None
        self._mutli_proc_end = None

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def sfreq(self):
        """
        Returns
        -------
        sampling_frequency: float, list
                            The sampling frequencies employed by the dataset.
        """
        raise NotImplementedError

    @property
    def channels(self):
        """
        Returns
        -------
        channels: list
                  The channel sets used by the dataset.
                """
        raise NotImplementedError

    @property
    def sequence_length(self):
        """
        Returns
        -------
        sequence_length: int, list
                         The length of each instance in number of samples
            """
        raise NotImplementedError

    def clone(self):
        """
        A copy of this object to allow the repetition of recordings, thinkers, etc. that load data from
        the same memory/files but have their own tracking of ids.

        Returns
        -------
        cloned : DN3ataset
                 New copy of this object.
        """
        return copy.deepcopy(self)

    def add_transform(self, transform):
        """
        Add a transformation that is applied to every fetched item in the dataset

        Parameters
        ----------
        transform : BaseTransform
                    For each item retrieved by __getitem__, transform is called to modify that item.
        """
        if isinstance(transform, InstanceTransform):
            self._transforms.append(transform)

    def _execute_transforms(self, *x):
        for transform in self._transforms:
            assert isinstance(transform, InstanceTransform)
            if transform.only_trial_data:
                new_x = transform(x[0])
                if isinstance(new_x, (list, tuple)):
                    x = (*new_x, *x[1:])
                else:
                    x = (new_x, *x[1:])
            else:
                x = transform(*x)

            if self._safe_mode:
                for i in range(len(x)):
                    if torch.any(torch.isnan(x[i])):
                        raise DN3atasetNanFound("NaN generated by transform {} for {}'th tensor".format(
                            self, i))
        return x

    def clear_transforms(self):
        """
        Remove all added transforms from dataset.
        """
        self._transforms = list()

    def preprocess(self, preprocessor: Preprocessor, apply_transform=True):
        """
        Applies a preprocessor to the dataset

        Parameters
        ----------
        preprocessor : Preprocessor
                       A preprocessor to be applied
        apply_transform : bool
                          Whether to apply the transform to this dataset (and all members e.g thinkers or sessions)
                          after preprocessing them. Alternatively, the preprocessor is returned for manual application
                          of its transform through :meth:`Preprocessor.get_transform()`

        Returns
        ---------
        processed_data : ndarry
                         Data that has been modified by the preprocessor, should be in the shape of [*, C, T], with C
                         and T according with the `channels` and `sequence_length` properties respectively.
        """
        raise NotImplementedError

    def to_numpy(self, batch_size=64, batch_transforms: list = None, num_workers=4, **dataloader_kwargs):
        """
        Commits the dataset to numpy-formatted arrays. Useful for saving dataset to disk, or preparing for tools that
        expect numpy-formatted data rather than iteratable.

        Notes
        -----
        A pytorch :any:`DataLoader` is used to fetch the data to conveniently leverage multiprocessing, and naturally

        Parameters
        ----------
        batch_size: int
                   The number of items to fetch per worker. This probably doesn't need much tuning.
        num_workers: int
                     The number of spawned processes to fetch and transform data.
        batch_transforms: list
                         These are potential batch-level transforms that
        dataloader_kwargs: dict
                          Keyword arguments for the pytorch :any:`DataLoader` that underpins the fetched data

        Returns
        -------
        data: list
              A list of numpy arrays.
        """
        dataloader_kwargs.setdefault('batch_size', batch_size)
        dataloader_kwargs.setdefault('num_workers', num_workers)
        dataloader_kwargs.setdefault('shuffle', False)
        dataloader_kwargs.setdefault('drop_last', False)

        batch_transforms = list() if batch_transforms is None else batch_transforms

        loaded = None
        loader = DataLoader(self, **dataloader_kwargs)
        for batch in tqdm.tqdm(loader, desc="Loading Batches"):
            for xform in batch_transforms:
                assert callable(xform)
                batch = xform(batch)
            # cpu just to be certain, shouldn't affect things otherwise
            batch = [b.cpu().numpy() for b in batch]
            if loaded is None:
                loaded = batch
            else:
                loaded = [np.concatenate([loaded[i], batch[i]], axis=0) for i in range(len(batch))]

        return loaded


class LabelSmoothedCrossEntropyLoss(torch.nn.Module):
    """this loss performs label smoothing to compute cross-entropy with soft labels, when smoothing=0.0, this
    is the same as torch.nn.CrossEntropyLoss"""

    def __init__(self, n_classes, smoothing=0.0, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.n_classes = n_classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class DN3BaseModel(nn.Module):
    """
    This is a base model used by the provided models in the library that is meant to make those included in this
    library as powerful and multi-purpose as is reasonable.

    It is not strictly necessary to have new modules inherit from this, any nn.Module should suffice, but it provides
    some integrated conveniences...

    The premise of this model is that deep learning models can be understood as *learned pipelines*. These
    :any:`DN3BaseModel` objects, are re-interpreted as a two-stage pipeline, the two stages being *feature extraction*
    and *classification*.
    """
    def __init__(self, samples, channels, return_features=True):
        super().__init__()
        self.samples = samples
        self.channels = channels
        self.return_features = return_features

    def forward(self, x):
        raise NotImplementedError

    def internal_loss(self, forward_pass_tensors):

        return None

    def clone(self):
        """
        This provides a standard way to copy models, weights and all.
        """
        return deepcopy(self)

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze

    @classmethod
    def from_dataset(cls, dataset: DN3ataset, **modelargs):
        print("Creating {} using: {} channels with trials of {} samples at {}Hz".format(cls.__name__,
                                                                                        len(dataset.channels),
                                                                                        dataset.sequence_length,
                                                                                        dataset.sfreq))
        assert isinstance(dataset, DN3ataset)
        return cls(samples=dataset.sequence_length, channels=len(dataset.channels), **modelargs)


class Classifier(DN3BaseModel):
    """
    A generic Classifer container. This container breaks operations up into feature extraction and feature
    classification to enable convenience in transfer learning and more.
    """

    @classmethod
    def from_dataset(cls, dataset: DN3ataset, **modelargs):
        """
        Create a classifier from a dataset.

        Parameters
        ----------
        dataset
        modelargs: dict
                   Options to construct the dataset, if dataset does not have listed targets, targets must be specified
                   in the keyword arguments or will fall back to 2.

        Returns
        -------
        model: Classifier
               A new `Classifier` ready to classifiy data from `dataset`
        """
        if hasattr(dataset, 'get_targets'):
            targets = len(np.unique(dataset.get_targets()))
        elif dataset.info is not None and isinstance(dataset.info.targets, int):
            targets = dataset.info.targets
        else:
            targets = 2
        modelargs.setdefault('targets', targets)
        print("Creating {} using: {} channels x {} samples at {}Hz | {} targets".format(cls.__name__,
                                                                                        len(dataset.channels),
                                                                                        dataset.sequence_length,
                                                                                        dataset.sfreq,
                                                                                        modelargs['targets']))
        assert isinstance(dataset, DN3ataset)
        return cls(samples=dataset.sequence_length, channels=len(dataset.channels), **modelargs)

    def __init__(self, targets, samples, channels, return_features=True):
        super(Classifier, self).__init__(samples, channels, return_features=return_features)
        self.targets = targets
        self.make_new_classification_layer()
        self._init_state = self.state_dict()

    def reset(self):
        self.load_state_dict(self._init_state)

    def forward(self, *x):
        features = self.features_forward(*x)
        if self.return_features:
            return self.classifier_forward(features), features
        else:
            return self.classifier_forward(features)

    def make_new_classification_layer(self):
        """
        This allows for a distinction between the classification layer(s) and the rest of the network. Using a basic
        formulation of a network being composed of two parts feature_extractor & classifier.

        This method is for implementing the classification side, so that methods like :py:meth:`freeze_features` works
        as intended.

        Anything besides a layer that just flattens anything incoming to a vector and Linearly weights this to the
        target should override this method, and there should be a variable called `self.classifier`

        """
        classifier = nn.Linear(self.num_features_for_classification, self.targets)
        nn.init.xavier_normal_(classifier.weight)
        classifier.bias.data.zero_()
        self.classifier = nn.Sequential(Flatten(), classifier)

    def freeze_features(self, unfreeze=False, freeze_classifier=False):
        """
        In many cases, the features learned by a model in one domain can be applied to another case.

        This method freezes (or un-freezes) all but the `classifier` layer. So that any further training does not (or
        does if unfreeze=True) affect these weights.

        Parameters
        ----------
        unfreeze : bool
                   To unfreeze weights after a previous call to this.
        freeze_classifier: bool
                   Commonly, the classifier layer will not be frozen (default). Setting this to `True` will freeze this
                   layer too.
        """
        super(Classifier, self).freeze_features(unfreeze=unfreeze)

        if isinstance(self.classifier, nn.Module) and not freeze_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = True

    @property
    def num_features_for_classification(self):
        raise NotImplementedError

    def classifier_forward(self, features):
        return self.classifier(features)

    def features_forward(self, x):
        raise NotImplementedError

    def load(self, filename, include_classifier=False, freeze_features=True):
        state_dict = torch.load(filename)
        if not include_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)
        if freeze_features:
            self.freeze_features()

    def save(self, filename, ignore_classifier=False):
        state_dict = self.state_dict()
        if ignore_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        print("Saving to {} ...".format(filename))
        torch.save(state_dict, filename)

class BatchTransform(object):

    def __init__(self, only_trial_data=True):
        """
        Batch transforms are operations that are performed on trial tensors after being accumulated into batches via the
        :meth:`__call__` method. Ideally this is implemented with pytorch operations for ease of execution graph
        integration.
        """
        self.only_trial_data = only_trial_data

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, *x, training=False):
        """
        Modifies a batch of tensors.

        Parameters
        ----------
        x : torch.Tensor, tuple
            A batch of trial instance tensor. If initialized with `only_trial_data=False`, then this includes batches
            of all other loaded tensors as well.
        training: bool
                  Indicates whether this is a training batch or otherwise, allowing for alternate behaviour during
                  evaluation.

        Returns
        -------
        x : torch.Tensor, tuple
            The modified trial tensor batch, or tensors if not `only_trial_data`
        """
        raise NotImplementedError()

def _make_span_from_seeds(seeds, span, total=None):
    inds = list()
    for seed in seeds:
        for i in range(seed, seed + span):
            if total is not None and i >= total:
                break
            elif i not in inds:
                inds.append(int(i))
    return np.array(inds)


def _make_mask(shape, p, total, span, allow_no_inds=False):
    # num_mask_spans = np.sum(np.random.rand(total) < p)
    # num_mask_spans = int(p * total)
    mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)

    for i in range(shape[0]):
        mask_seeds = list()
        while not allow_no_inds and len(mask_seeds) == 0 and p > 0:
            mask_seeds = np.nonzero(np.random.rand(total) < p)[0]

        mask[i, _make_span_from_seeds(mask_seeds, span, total=total)] = True

    return mask

class BaseProcess(object):
    """
    Initialization of the Base Trainable object. Any learning procedure that leverages DN3atasets should subclass
    this base class.

    By default uses the SGD with momentum optimization.
    """

    def __init__(self, lr=0.001, metrics=None, evaluation_only_metrics=None, l2_weight_decay=0.01, cuda=None, **kwargs):
        """
        Initialization of the Base Trainable object. Any learning procedure that leverages DN3atasets should subclass
        this base class.

        By default uses the SGD with momentum optimization.

        Parameters
        ----------
        cuda : bool, string, None
               If boolean, sets whether to enable training on the GPU, if a string, specifies can be used to specify
               which device to use. If None (default) figures it out automatically.
        lr : float
             The learning rate to use, this will probably something that should be tuned for each application.
             Start with multiplying or dividing by values of 2, 5 or 10 to seek out a good number.
        metrics : dict, list
                  A dictionary of named (keys) metrics (values) or some iterable set of metrics that will be identified
                  by their class names.
        evaluation_only_metrics : list
                                 A list of names of metrics that will be used for evaluation only (not calculated or
                                 reported during training steps).
        l2_weight_decay : float
                          One of the simplest and most common regularizing techniques. If you find a model rapidly
                          reaching high training accuracy (and not validation) increase this. If having trouble fitting
                          the training data, decrease this.
        kwargs : dict
                 Arguments that will be used by the processes' :py:meth:`BaseProcess.build_network()` method.
        """
        if cuda is None:
            cuda = torch.cuda.is_available()
            if cuda:
                tqdm.tqdm.write("GPU(s) detected: training and model execution will be performed on GPU.")
        if isinstance(cuda, bool):
            cuda = "cuda" if cuda else "cpu"
        assert isinstance(cuda, str)
        self.cuda = cuda
        self.device = torch.device(cuda)
        self._eval_metrics = list() if evaluation_only_metrics is None else list(evaluation_only_metrics).copy()
        self.metrics = OrderedDict()
        if metrics is not None:
            if isinstance(metrics, (list, tuple)):
                metrics = {m.__class__.__name__: m for m in metrics}
            if isinstance(metrics, dict):
                self.add_metrics(metrics)

        _before_members = set(self.__dict__.keys())
        self.build_network(**kwargs)
        new_members = set(self.__dict__.keys()).difference(_before_members)
        self._training = False
        self._trainables = list()
        for member in new_members:
            if isinstance(self.__dict__[member], (torch.nn.Module, torch.Tensor)):
                if not (isinstance(self.__dict__[member], torch.Tensor) and not self.__dict__[member].requires_grad):
                    self._trainables.append(member)
                self.__dict__[member] = self.__dict__[member].to(self.device)

        self.optimizer = torch.optim.SGD(self.parameters(), weight_decay=l2_weight_decay, lr=lr, nesterov=True,
                                         momentum=0.9)
        self.scheduler = None
        self.scheduler_after_batch = False
        self.epoch = None
        self.lr = lr
        self.weight_decay = l2_weight_decay

        self._batch_transforms = list()
        self._eval_transforms = list()

    def set_optimizer(self, optimizer):
        assert isinstance(optimizer, torch.optim.Optimizer)
        del self.optimizer
        self.optimizer = optimizer
        self.lr = float(self.optimizer.param_groups[0]['lr'])

    def set_scheduler(self, scheduler, step_every_batch=False):
        """
        This allow the addition of a learning rate schedule to the process. By default, a linear warmup with cosine
        decay will be used. Any scheduler that is an instance of :any:`Scheduler` (pytorch's schedulers, or extensions
        thereof) can be set here. Additionally, a string keywords can be used including:
          - "constant"

        Parameters
        ----------
        scheduler: str, Scheduler
        step_every_batch: bool
                          Whether to call step after every batch (if `True`), or after every epoch (`False`)

        """
        if isinstance(scheduler, str):
            if scheduler.lower() == 'constant':
                scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda e: 1.0)
            else:
                raise ValueError("Scheduler {} is not supported.".format(scheduler))
        # This is the most common one that needs this, force this to be true
        elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            self.scheduler_after_batch = True
        else:
            self.scheduler_after_batch = step_every_batch
        self.scheduler = scheduler

    def add_metrics(self, metrics: dict, evaluation_only=False):
        self.metrics.update(**metrics)
        if evaluation_only:
            self._eval_metrics += list(metrics.keys())

    def _optimize_dataloader_kwargs(self, num_worker_cap=6, **loader_kwargs):
        loader_kwargs.setdefault('pin_memory', self.cuda == 'cuda')
        # Use multiple worker processes when NOT DEBUGGING
        if gettrace() is None:
            try:
                # Find number of cpus available (taken from second answer):
                # https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python
                m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                              open('/proc/self/status').read())
                nw = bin(int(m.group(1).replace(',', ''), 16)).count('1')
                # Cap the number of workers at 6 (actually 4) to avoid pummeling disks too hard
                nw = min(num_worker_cap, nw)
            except FileNotFoundError:
                # Fallback for when proc/self/status does not exist
                nw = 2
        else:
            # 0 workers means not extra processes are spun up
            nw = 2
        loader_kwargs.setdefault('num_workers', int(nw - 2))
        print("Loading data with {} additional workers".format(loader_kwargs['num_workers']))
        return loader_kwargs

    def _get_batch(self, iterator):
        batch = [x.to(self.device, non_blocking=self.cuda == 'cuda') for x in next(iterator)]
        xforms = self._batch_transforms if self._training else self._eval_transforms
        for xform in xforms:
            if xform.only_trial_data:
                batch[0] = xform(batch[0])
            else:
                batch = xform(batch)
        return batch

    def add_batch_transform(self, transform: BatchTransform, training_only=True):
        self._batch_transforms.append(transform)
        if not training_only:
            self._eval_transforms.append(transform)

    def clear_batch_transforms(self):
        self._batch_transforms = list()
        self._eval_transforms = list()

    def build_network(self, **kwargs):
        """
        This method is used to add trainable modules to the process. Rather than placing objects for training
        in the __init__ method, they should be placed here.

        By default any arguments that propagate unused from __init__ are included here.
        """
        self.__dict__.update(**kwargs)

    def parameters(self):
        """
        All the trainable parameters in the Trainable. This includes any architecture parameters and meta-parameters.

        Returns
        -------
        params :
                 An iterator of parameters
        """
        for member in self._trainables:
            yield from self.__dict__[member].parameters()

    def forward(self, *inputs):
        """
        Given a batch of inputs, return the outputs produced by the trainable module.

        Parameters
        ----------
        inputs :
               Tensors needed for underlying module.

        Returns
        -------
        outputs :
                Outputs of module

        """
        raise NotImplementedError

    def calculate_loss(self, inputs, outputs):
        """
        Given the inputs to and outputs from underlying modules, calculate the loss.

        Returns
        -------
        Loss :
             Single loss quantity to be minimized.
        """
        if isinstance(outputs, (tuple, list)):
            device = outputs[0].device
        else:
            device = outputs.device
        loss_fn = self.loss
        if hasattr(self.loss, 'to'):
            loss_fn = loss_fn.to(device)
        return loss_fn(outputs, inputs[-1])

    def calculate_metrics(self, inputs, outputs):
        """
        Given the inputs to and outputs from the underlying module. Return tracked metrics.

        Parameters
        ----------
        inputs :
               Input tensors.
        outputs :
                Output tensors.

        Returns
        -------
        metrics : OrderedDict, None
                  Dictionary of metric quantities.
        """
        metrics = OrderedDict()
        for met_name, met_fn in self.metrics.items():
            if self._training and met_name in self._eval_metrics:
                continue
            try:
                metrics[met_name] = met_fn(inputs, outputs)
            # I know its super broad, but basically if metrics fail during training, I want to just ignore them...
            except:
                continue
        return metrics

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

    def train(self, mode=True):
        self._training = mode
        for member in self._trainables:
            self.__dict__[member].train(mode=mode)

    def train_step(self, *inputs):
        self.train(True)
        outputs = self.forward(*inputs)
        loss = self.calculate_loss(inputs, outputs)
        self.backward(loss)

        self.optimizer.step()
        if self.scheduler is not None and self.scheduler_after_batch:
            self.scheduler.step()

        train_metrics = self.calculate_metrics(inputs, outputs)
        train_metrics.setdefault('loss', loss.item())

        return train_metrics

    def evaluate(self, dataset, **loader_kwargs):
        """
        Calculate and return metrics for a dataset

        Parameters
        ----------
        dataset: DN3ataset, DataLoader
                 The dataset that will be used for evaluation, if not a DataLoader, one will be constructed
        loader_kwargs: dict
                       Args that will be passed to the dataloader, but `shuffle` and `drop_last` will be both be
                       forced to `False`

        Returns
        -------
        metrics : OrderedDict
                Metric scores for the entire
        """
        self.train(False)
        inputs, outputs = self.predict(dataset, **loader_kwargs)
        metrics = self.calculate_metrics(inputs, outputs)
        metrics['loss'] = self.calculate_loss(inputs, outputs).item()
        return metrics

    def predict(self, dataset, **loader_kwargs):
        """
        Determine the outputs for all loaded data from the dataset

        Parameters
        ----------
        dataset: DN3ataset, DataLoader
                 The dataset that will be used for evaluation, if not a DataLoader, one will be constructed
        loader_kwargs: dict
                       Args that will be passed to the dataloader, but `shuffle` and `drop_last` will be both be
                       forced to `False`

        Returns
        -------
        inputs : Tensor
                 The exact inputs used to calculate the outputs (in case they were stochastic and need saving)
        outputs : Tensor
                  The outputs from each run of :function:`forward`
        """
        self.train(False)
        loader_kwargs.setdefault('batch_size', 1)
        dataset = self._make_dataloader(dataset, **loader_kwargs)

        pbar = tqdm.trange(len(dataset), desc="Predicting")
        data_iterator = iter(dataset)

        inputs = list()
        outputs = list()

        with torch.no_grad():
            for iteration in pbar:
                input_batch = self._get_batch(data_iterator)
                output_batch = self.forward(*input_batch)

                inputs.append([tensor.cpu() for tensor in input_batch])
                if isinstance(output_batch, torch.Tensor):
                    outputs.append(output_batch.cpu())
                else:
                    outputs.append([tensor.cpu() for tensor in output_batch])

        def package_multiple_tensors(batches: list):
            if isinstance(batches[0], torch.Tensor):
                return torch.cat(batches)
            elif isinstance(batches[0], (tuple, list)):
                return [torch.cat(b) for b in zip(*batches)]

        return package_multiple_tensors(inputs), package_multiple_tensors(outputs)

    @classmethod
    def standard_logging(cls, metrics: dict, start_message="End of Epoch"):
        if start_message.rstrip()[-1] != '|':
            start_message = start_message.rstrip() + " |"
        for m in metrics:
            if 'acc' in m.lower() or 'pct' in m.lower():
                start_message += " {}: {:.2%} |".format(m, metrics[m])
            elif m == 'lr':
                start_message += " {}: {:.3e} |".format(m, metrics[m])
            else:
                start_message += " {}: {:.3f} |".format(m, metrics[m])
        tqdm.tqdm.write(start_message)

    def save_best(self):
        """
        Create a snapshot of what is being currently trained for re-laoding with the :py:meth:`load_best()` method.

        Returns
        -------
        best : Any
               Whatever format is needed for :py:meth:`load_best()`, will be the argument provided to it.
        """
        return [{k: v.cpu() for k, v in self.__dict__[m].state_dict().items()} for m in self._trainables]

    def load_best(self, best):
        """
        Load the parameters as saved by :py:meth:`save_best()`.

        Parameters
        ----------
        best: Any
        """
        for m, state_dict in zip(self._trainables, best):
            self.__dict__[m].load_state_dict({k: v.to(self.device) for k, v in state_dict.items()})

    def _retain_best(self, old_checkpoint, metrics_to_check: dict, retain_string: str):
        if retain_string is None:
            return old_checkpoint
        best_checkpoint = old_checkpoint

        def found_best():
            tqdm.tqdm.write("Best {}. Retaining checkpoint...".format(retain_string))
            self.best_metric = metrics_to_check[retain_string]
            return self.save_best()

        if retain_string not in metrics_to_check.keys():
            tqdm.tqdm.write("No metric {} found in recorded metrics. Not saving best.")
        if self.best_metric is None:
            best_checkpoint = found_best()
        elif retain_string == 'loss' and metrics_to_check[retain_string] <= self.best_metric:
            best_checkpoint = found_best()
        elif retain_string != 'loss' and metrics_to_check[retain_string] >= self.best_metric:
            best_checkpoint = found_best()

        return best_checkpoint

    @staticmethod
    def _dataloader_args(dataset, training=False, **loader_kwargs):
        # Only shuffle and drop last when training
        loader_kwargs.setdefault('shuffle', training)
        loader_kwargs.setdefault('drop_last', training)

        return loader_kwargs

    def _make_dataloader(self, dataset, training=False, **loader_kwargs):
        """Any args that make more sense as a convenience function to be set"""
        if isinstance(dataset, DataLoader):
            return dataset

        return DataLoader(dataset, **self._dataloader_args(dataset, training, **loader_kwargs))

    def fit(self, training_dataset, epochs=1, validation_dataset=None, step_callback=None,
            resume_epoch=None, resume_iteration=None, log_callback=None, validation_callback=None,
            epoch_callback=None, batch_size=8, warmup_frac=0.2, retain_best='loss',
            validation_interval=None, train_log_interval=None, **loader_kwargs):
        """
        sklearn/keras-like convenience method to simply proceed with training across multiple epochs of the provided
        dataset

        Parameters
        ----------
        training_dataset : DN3ataset, DataLoader
        validation_dataset : DN3ataset, DataLoader
        epochs : int
                 Total number of epochs to fit
        resume_epoch : int
                      The starting epoch to train from. This will likely only be used to resume training at a certain
                      point.
        resume_iteration : int
                          Similar to start epoch but specified in batches. This can either be used alone, or in
                          conjunction with `start_epoch`. If used alone, the start epoch is the floor of
                          `start_iteration` divided by batches per epoch. In other words this specifies cumulative
                          batches if start_epoch is not specified, and relative to the current epoch otherwise.
        step_callback : callable
                        Function to run after every training step that has signature: fn(train_metrics) -> None
        log_callback : callable
                       Function to run after every log interval that has signature: fn(train_metrics) -> None
        validation_callback : callable
                        Function to run after every time the validation dataset is run through. This typically has the
                        result of this and the `epoch_callback` called at the end of the epoch, but this is also called
                        after `validation_interval` batches.
                        This callback has the signature: fn(validation_metrics) -> None
        epoch_callback : callable
                        Function to run after every epoch that has signature: fn(validation_metrics) -> None
        batch_size : int
                     The batch_size to be used for the training and validation datasets. This is ignored if they are
                     provided as `DataLoader`.
        warmup_frac : float
                      The fraction of iterations that will be spent *increasing* the learning rate under the default
                      1cycle policy (with cosine annealing). Value will be automatically clamped values between [0, 0.5]
        retain_best : (str, None)
                      **If `validation_dataset` is provided**, which model weights to retain. If 'loss' (default), will
                      retain the model at the epoch with the lowest validation loss. If another string, will assume that
                      is the metric to monitor for the *highest score*. If None, the final model is used.
        validation_interval: int, None
                             The number of batches between checking the validation dataset
        train_log_interval: int, None
                      The number of batches between persistent logging of training metrics, if None (default) happens
                      at the end of every epoch.
        loader_kwargs :
                      Any remaining keyword arguments will be passed as such to any DataLoaders that are automatically
                      constructed. If both training and validation datasets are provided as `DataLoaders`, this will be
                      ignored.

        Notes
        -----
        If the datasets above are provided as DN3atasets, automatic optimizations are performed to speed up loading.
        These include setting the number of workers = to the number of CPUs/system threads - 1, and pinning memory for
        rapid CUDA transfer if leveraging the GPU. Unless you are very comfortable with PyTorch, it's probably better
        to not provide your own DataLoader, and let this be done automatically.

        Returns
        -------
        train_log : Dataframe
                    Metrics after each iteration of training as a pandas dataframe
        validation_log : Dataframe
                         Validation metrics after each epoch of training as a pandas dataframe
        """
        loader_kwargs.setdefault('batch_size', batch_size)
        loader_kwargs = self._optimize_dataloader_kwargs(**loader_kwargs)
        training_dataset = self._make_dataloader(training_dataset, training=True, **loader_kwargs)

        if resume_epoch is None:
            if resume_iteration is None or resume_iteration < len(training_dataset):
                resume_epoch = 1
            else:
                resume_epoch = resume_iteration // len(training_dataset)
        resume_iteration = 1 if resume_iteration is None else resume_iteration % len(training_dataset)

        _clear_scheduler_after = self.scheduler is None
        if _clear_scheduler_after:
            last_epoch_workaround = len(training_dataset) * (resume_epoch - 1) + resume_iteration
            last_epoch_workaround = -1 if last_epoch_workaround <= 1 else last_epoch_workaround
            self.set_scheduler(
                torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.lr, epochs=epochs,
                                                    steps_per_epoch=len(training_dataset),
                                                    pct_start=warmup_frac,
                                                    last_epoch=last_epoch_workaround)
            )

        validation_log = list()
        train_log = list()
        self.best_metric = None
        best_model = self.save_best()

        train_log_interval = len(training_dataset) if train_log_interval is None else train_log_interval
        metrics = OrderedDict()

        def update_metrics(new_metrics: dict, iterations):
            if len(metrics) == 0:
                return metrics.update(new_metrics)
            else:
                for m in new_metrics:
                    try:
                        metrics[m] = (metrics[m] * (iterations - 1) + new_metrics[m]) / iterations
                    except KeyError:
                        metrics[m] = new_metrics[m]

        def print_training_metrics(epoch, iteration=None):
            if iteration is not None:
                self.standard_logging(metrics, "Training: Epoch {} - Iteration {}".format(epoch, iteration))
            else:
                self.standard_logging(metrics, "Training: End of Epoch {}".format(epoch))

        def _validation(epoch, iteration=None):
            _metrics = self.evaluate(validation_dataset, **loader_kwargs)
            if iteration is not None:
                self.standard_logging(_metrics, "Validation: Epoch {} - Iteration {}".format(epoch, iteration))
            else:
                self.standard_logging(_metrics, "Validation: End of Epoch {}".format(epoch))
            _metrics['epoch'] = epoch
            validation_log.append(_metrics)
            if callable(validation_callback):
                validation_callback(_metrics)
            return _metrics

        epoch_bar = tqdm.trange(resume_epoch, epochs + 1, desc="Epoch", unit='epoch', initial=resume_epoch, total=epochs)
        for epoch in epoch_bar:
            self.epoch = epoch
            pbar = tqdm.trange(resume_iteration, len(training_dataset) + 1, desc="Iteration", unit='batches',
                               initial=resume_iteration, total=len(training_dataset))
            data_iterator = iter(training_dataset)
            for iteration in pbar:
                inputs = self._get_batch(data_iterator)
                train_metrics = self.train_step(*inputs)
                train_metrics['lr'] = self.optimizer.param_groups[0]['lr']
                if 'momentum' in self.optimizer.defaults:
                    train_metrics['momentum'] = self.optimizer.param_groups[0]['momentum']
                update_metrics(train_metrics, iteration+1)
                pbar.set_postfix(metrics)
                train_metrics['epoch'] = epoch
                train_metrics['iteration'] = iteration
                train_log.append(train_metrics)
                if callable(step_callback):
                    step_callback(train_metrics)

                if iteration % train_log_interval == 0 and pbar.total != iteration:
                    print_training_metrics(epoch, iteration)
                    train_metrics['epoch'] = epoch
                    train_metrics['iteration'] = iteration
                    if callable(log_callback):
                        log_callback(metrics)
                    metrics = OrderedDict()

                if isinstance(validation_interval, int) and (iteration % validation_interval == 0)\
                        and validation_dataset is not None:
                    _m = _validation(epoch, iteration)
                    best_model = self._retain_best(best_model, _m, retain_best)

            # Make epoch summary
            metrics = DataFrame(train_log)
            metrics = metrics[metrics['epoch'] == epoch]
            metrics = metrics.mean().to_dict()
            metrics.pop('iteration', None)
            print_training_metrics(epoch)

            if validation_dataset is not None:
                metrics = _validation(epoch)
                best_model = self._retain_best(best_model, metrics, retain_best)

            if callable(epoch_callback):
                epoch_callback(metrics)
            metrics = OrderedDict()
            # All future epochs should not start offset in iterations
            resume_iteration = 1

            if not self.scheduler_after_batch and self.scheduler is not None:
                tqdm.tqdm.write(f"Step {self.scheduler.get_last_lr()} {self.scheduler.last_epoch}")
                self.scheduler.step()

        if _clear_scheduler_after:
            self.set_scheduler(None)
        self.epoch = None

        if retain_best is not None and validation_dataset is not None:
            tqdm.tqdm.write("Loading best model...")
            self.load_best(best_model)

        return DataFrame(train_log), DataFrame(validation_log)


class StandardClassification(BaseProcess):

    def __init__(self, classifier: torch.nn.Module, loss_fn=None, cuda=None, metrics=None, learning_rate=0.01,
                 label_smoothing=None, **kwargs):
        if isinstance(metrics, dict):
            metrics.setdefault('Accuracy', self._simple_accuracy)
        else:
            metrics = dict(Accuracy=self._simple_accuracy)
        super(StandardClassification, self).__init__(cuda=cuda, lr=learning_rate, classifier=classifier,
                                                     metrics=metrics, **kwargs)
        if label_smoothing is not None and isinstance(label_smoothing, float) and (0 < label_smoothing < 1):
            self.loss = LabelSmoothedCrossEntropyLoss(self.classifier.targets, smoothing=label_smoothing).\
                to(self.device)
        elif loss_fn is None:
            self.loss = torch.nn.CrossEntropyLoss().to(self.device)
        else:
            self.loss = loss_fn.to(self.device)
        self.best_metric = None

    @staticmethod
    def _simple_accuracy(inputs, outputs):
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        # average over last dimensions
        while len(outputs.shape) >= 3:
            outputs = outputs.mean(dim=-1)
        return (inputs[-1] == outputs.argmax(dim=-1)).float().mean().item()

    def forward(self, *inputs):
        if isinstance(self.classifier, Classifier) and self.classifier.return_features:
            prediction, _ = self.classifier(*inputs[:-1])
        else:
            prediction = self.classifier(*inputs[:-1])
        return prediction

    def calculate_loss(self, inputs, outputs):
        inputs = list(inputs)

        def expand_for_strided_loss(factors):
            inputs[-1] = inputs[-1].unsqueeze(-1).expand(-1, *factors)

        check_me = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        if len(check_me.shape) >= 3:
            expand_for_strided_loss(check_me.shape[2:])

        return super(StandardClassification, self).calculate_loss(inputs, outputs)

    def fit(self, training_dataset, epochs=1, validation_dataset=None, step_callback=None, epoch_callback=None,
            batch_size=8, warmup_frac=0.2, retain_best='loss', balance_method=None, **loader_kwargs):
        """
        sklearn/keras-like convenience method to simply proceed with training across multiple epochs of the provided
        dataset

        Parameters
        ----------
        training_dataset : DN3ataset, DataLoader
        validation_dataset : DN3ataset, DataLoader
        epochs : int
        step_callback : callable
                        Function to run after every training step that has signature: fn(train_metrics) -> None
        epoch_callback : callable
                        Function to run after every epoch that has signature: fn(validation_metrics) -> None
        batch_size : int
                     The batch_size to be used for the training and validation datasets. This is ignored if they are
                     provided as `DataLoader`.
        warmup_frac : float
                      The fraction of iterations that will be spent *increasing* the learning rate under the default
                      1cycle policy (with cosine annealing). Value will be automatically clamped values between [0, 0.5]
        retain_best : (str, None)
                      **If `validation_dataset` is provided**, which model weights to retain. If 'loss' (default), will
                      retain the model at the epoch with the lowest validation loss. If another string, will assume that
                      is the metric to monitor for the *highest score*. If None, the final model is used.
        balance_method : (None, str)
                         If and how to balance training samples when training. `None` (default) will simply randomly
                         sample all training samples equally. 'undersample' will sample each class N_min times
                         where N_min is equal to the number of examples in the minority class. 'oversample' will sample
                         each class N_max times, where N_max is the number of the majority class.
        loader_kwargs :
                      Any remaining keyword arguments will be passed as such to any DataLoaders that are automatically
                      constructed. If both training and validation datasets are provided as `DataLoaders`, this will be
                      ignored.

        Notes
        -----
        If the datasets above are provided as DN3atasets, automatic optimizations are performed to speed up loading.
        These include setting the number of workers = to the number of CPUs/system threads - 1, and pinning memory for
        rapid CUDA transfer if leveraging the GPU. Unless you are very comfortable with PyTorch, it's probably better
        to not provide your own DataLoader, and let this be done automatically.

        Returns
        -------
        train_log : Dataframe
                    Metrics after each iteration of training as a pandas dataframe
        validation_log : Dataframe
                         Validation metrics after each epoch of training as a pandas dataframe
        """
        return super(StandardClassification, self).fit(training_dataset, epochs=epochs, step_callback=step_callback,
                                                       epoch_callback=epoch_callback, batch_size=batch_size,
                                                       warmup_frac=warmup_frac, retain_best=retain_best,
                                                       validation_dataset=validation_dataset,
                                                       balance_method=balance_method,
                                                       **loader_kwargs)

    BALANCE_METHODS = ['undersample', 'oversample', 'ldam']
    def _make_dataloader(self, dataset, training=False, **loader_kwargs):
        if isinstance(dataset, DataLoader):
            return dataset

        loader_kwargs = self._dataloader_args(dataset, training=training, **loader_kwargs)

        if training and loader_kwargs.get('sampler', None) is None and loader_kwargs.get('balance_method', None) \
                is not None:
            method = loader_kwargs.pop('balance_method')
            assert method.lower() in self.BALANCE_METHODS
            if not hasattr(dataset, 'get_targets'):
                print("Failed to create dataloader with {} balancing. {} does not support `get_targets()`.".format(
                    method, dataset
                ))
            elif method.lower() != 'ldam':
                sampler = balanced_undersampling(dataset) if method.lower() == 'undersample' \
                    else balanced_oversampling(dataset)
                # Shuffle is implied by the balanced sampling
                # loader_kwargs['shuffle'] = None
                loader_kwargs['sampler'] = sampler
            else:
                self.loss = create_ldam_loss(dataset)

        if loader_kwargs.get('sampler', None) is not None:
            loader_kwargs['shuffle'] = None

        # Make sure balance method is not passed to DataLoader at this point.
        loader_kwargs.pop('balance_method', None)

        return DataLoader(dataset, **loader_kwargs)


def get_label_balance(dataset):
    """
    Given a dataset, return the proportion of each target class and the counts of each class type

    Parameters
    ----------
    dataset

    Returns
    -------
    sample_weights, counts
    """
    assert hasattr(dataset, 'get_targets')
    labels = dataset.get_targets()
    counts = np.bincount(labels)
    train_weights = 1. / torch.tensor(counts, dtype=torch.float)
    sample_weights = train_weights[labels]
    class_freq = counts/counts.sum()
    if len(counts) < 10:
        tqdm.tqdm.write('Class frequency: {}'.format(' | '.join('{:.2f}'.format(c) for c in class_freq)))
    else:
        tqdm.tqdm.write("Class frequencies range from {:.2e} to {:.2e}".format(class_freq.min(), class_freq.max()))
    return sample_weights, counts


def balanced_undersampling(dataset, replacement=False):
    tqdm.tqdm.write("Undersampling for balanced distribution.")
    sample_weights, counts = get_label_balance(dataset)
    return WeightedRandomSampler(sample_weights, len(counts) * int(counts.min()), replacement=replacement)


def balanced_oversampling(dataset, replacement=True):
    tqdm.tqdm.write("Oversampling for balanced distribution.")
    sample_weights, counts = get_label_balance(dataset)
    return WeightedRandomSampler(sample_weights, len(counts) * int(counts.max()), replacement=replacement)


class LDAMLoss(torch.nn.Module):
    # September 2020 - Originally taken from: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
    # October   2020 - Modified to support non-cuda devices and a switch to activate drw

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        self._cls_nums = cls_num_list
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def _determine_drw_weights(self, beta=0.9999):
        effective_num = 1.0 - np.power(beta, self._cls_nums)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        return torch.from_numpy(per_cls_weights / np.sum(per_cls_weights) * len(self._cls_nums)).float()

    def drw(self, on=True, beta=0.9999):
        self.weight = self._determine_drw_weights(beta=beta) if on else None

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :].to(index.device), index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        w = self.weight.to(index.device) if self.weight is not None else None
        return torch.nn.functional.cross_entropy(self.s * output, target, weight=w)


def create_ldam_loss(training_dataset):
    sample_weights, counts = get_label_balance(training_dataset)
    return LDAMLoss(counts)


def _make_span_from_seeds(seeds, span, total=None):
    inds = list()
    for seed in seeds:
        for i in range(seed, seed + span):
            if total is not None and i >= total:
                break
            elif i not in inds:
                inds.append(int(i))
    return np.array(inds)


class BendingCollegeWav2Vec(BaseProcess):
    """
    A more wav2vec 2.0 style of constrastive self-supervision, more inspired-by than exactly like it.
    """
    def __init__(self, encoder, context_fn, mask_rate=0.1, mask_span=6, learning_rate=0.01, temp=0.5,
                 permuted_encodings=False, permuted_contexts=False, enc_feat_l2=0.001, multi_gpu=False,
                 l2_weight_decay=1e-4, unmasked_negative_frac=0.25, encoder_grad_frac=1.0,
                 num_negatives=100, **kwargs):
        self.predict_length = mask_span
        self._enc_downsample = encoder.downsampling_factor
        if multi_gpu:
            encoder = torch.nn.DataParallel(encoder)
            context_fn = torch.nn.DataParallel(context_fn)
        if encoder_grad_frac < 1:
            encoder.register_backward_hook(lambda module, in_grad, out_grad:
                                           tuple(encoder_grad_frac * ig for ig in in_grad))
        super(BendingCollegeWav2Vec, self).__init__(encoder=encoder, context_fn=context_fn,
                                                    loss_fn=torch.nn.CrossEntropyLoss(), lr=learning_rate,
                                                    l2_weight_decay=l2_weight_decay,
                                                    metrics=dict(Accuracy=self._contrastive_accuracy,
                                                                 Mask_pct=self._mask_pct), **kwargs)
        self.best_metric = None
        self.mask_rate = mask_rate
        self.mask_span = mask_span
        self.temp = temp
        self.permuted_encodings = permuted_encodings
        self.permuted_contexts = permuted_contexts
        self.beta = enc_feat_l2
        self.start_token = getattr(context_fn, 'start_token', None)
        self.unmasked_negative_frac = unmasked_negative_frac
        self.num_negatives = num_negatives

    def description(self, sequence_len):
        encoded_samples = self._enc_downsample(sequence_len)
        desc = "{} samples | mask span of {} at a rate of {} => E[masked] ~= {}".format(
            encoded_samples, self.mask_span, self.mask_rate,
            int(encoded_samples * self.mask_rate * self.mask_span))
        return desc

    def _generate_negatives(self, z):
        """Generate negative samples to compare each sequence location against"""
        batch_size, feat, full_len = z.shape
        z_k = z.permute([0, 2, 1]).reshape(-1, feat)
        negative_inds = torch.empty(batch_size, full_len, self.num_negatives).long()
        ind_weights = torch.ones(full_len, full_len) - torch.eye(full_len)
        with torch.no_grad():
            # candidates = torch.arange(full_len).unsqueeze(-1).expand(-1, self.num_negatives).flatten()
            for i in range(batch_size):
                negative_inds[i] = torch.multinomial(ind_weights, self.num_negatives) + i*full_len
            # From wav2vec 2.0 implementation, I don't understand
            # negative_inds[negative_inds >= candidates] += 1

        z_k = z_k[negative_inds.view(-1)].view(batch_size, full_len, self.num_negatives, feat)
        return z_k, negative_inds

    def _calculate_similarity(self, z, c, negatives):
        c = c[..., 1:].permute([0, 2, 1]).unsqueeze(-2)
        z = z.permute([0, 2, 1]).unsqueeze(-2)

        # In case the contextualizer matches exactly, need to avoid divide by zero errors
        negative_in_target = (c == negatives).all(-1)
        targets = torch.cat([z, negatives], dim=-2)

        logits = torch.nn.functional.cosine_similarity(c, targets, dim=-1) / self.temp
        if negative_in_target.any():
            logits[1:][negative_in_target] = float("-inf")

        return logits.view(-1, logits.shape[-1])

    def forward(self, *inputs):
        z = self.encoder(inputs[0])

        if self.permuted_encodings:
            z = z.permute([1, 2, 0])

        unmasked_z = z.clone()

        batch_size, feat, samples = z.shape

        if self._training:
            mask = _make_mask((batch_size, samples), self.mask_rate, samples, self.mask_span)
        else:
            mask = torch.zeros((batch_size, samples), requires_grad=False, dtype=torch.bool)
            half_avg_num_seeds = max(1, int(samples * self.mask_rate * 0.5))
            if samples <= self.mask_span * half_avg_num_seeds:
                raise ValueError("Masking the entire span, pointless.")
            mask[:, _make_span_from_seeds((samples // half_avg_num_seeds) * np.arange(half_avg_num_seeds).astype(int),
                                              self.mask_span)] = True

        c = self.context_fn(z, mask)

        # Select negative candidates and generate labels for which are correct labels
        negatives, negative_inds = self._generate_negatives(unmasked_z)

        # Prediction -> batch_size x predict_length x predict_length
        logits = self._calculate_similarity(unmasked_z, c, negatives)
        return logits, unmasked_z, mask, c

    @staticmethod
    def _mask_pct(inputs, outputs):
        return outputs[2].float().mean().item()

    @staticmethod
    def _contrastive_accuracy(inputs, outputs):
        logits = outputs[0]
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        return StandardClassification._simple_accuracy([labels], logits)

    def calculate_loss(self, inputs, outputs):
        logits = outputs[0]
        # The 0'th index is the correct position
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)

        # Note that loss_fn here integrates the softmax as per the normal classification pipeline (leveraging logsumexp)
        return self.loss_fn(logits, labels) + self.beta * outputs[1].pow(2).mean()


class BendingCollegeClassification(BendingCollegeWav2Vec, StandardClassification):

    def __init__(self, bendr_model, mask_rate=0.1, mask_span=6, learning_rate=0.01, temp=0.5,
                 permuted_encodings=False, permuted_contexts=False, enc_feat_l2=0.001, multi_gpu=False,
                 l2_weight_decay=1e-4, unmasked_negative_frac=0.25, encoder_grad_frac=1.0,
                 num_negatives=100, max_reconstruction_loss_frac=0.2, **kwargs):
        StandardClassification.__init__(self, bendr_model.classifier,
                                        metrics={
                                            'Accuracy': lambda i, o: self._simple_accuracy(i, o[-1]),
                                            'Contrast-Accuracy':self._contrastive_accuracy,
                                            'Mask-pct':self._mask_pct
                                        },
                                        encoder=bendr_model.encoder,
                                        context_fn=bendr_model.contextualizer)
        if isinstance(bendr_model.encoder, torch.nn.DataParallel):
            encoder = bendr_model.encoder.module
            contextualizer = bendr_model.contextualizer.module
        else:
            encoder = bendr_model.encoder
            contextualizer = bendr_model.contextualizer

        self.predict_length = mask_span
        self._enc_downsample = encoder.downsampling_factor
        if encoder_grad_frac < 1:
            encoder.register_backward_hook(lambda module, in_grad, out_grad:
                                                       tuple(encoder_grad_frac * ig for ig in in_grad))
        self.best_metric = None
        self.mask_rate = mask_rate
        self.mask_span = mask_span
        self.temp = temp
        self.permuted_encodings = permuted_encodings
        self.permuted_contexts = permuted_contexts
        self.beta = enc_feat_l2
        self.start_token = getattr(contextualizer, 'start_token', None)
        self.unmasked_negative_frac = unmasked_negative_frac
        self.num_negatives = num_negatives
        self.r_lambda = max_reconstruction_loss_frac

    def forward(self, *inputs):
        logits, unmasked_z, mask, c = BendingCollegeWav2Vec.forward(self, *inputs)
        prediction = self.classifier(c[..., 0])
        return logits, unmasked_z, mask, prediction

    def calculate_loss(self, inputs, outputs):
        logits = outputs[0]
        # The 0'th index is the correct position
        correct_idx = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)

        mlm_loss = self.loss(logits, correct_idx) + self.beta * outputs[1].pow(2).mean()
        cls_loss = StandardClassification.calculate_loss(self, inputs, outputs[-1])

        return (1 - self.r_lambda) * cls_loss + self.r_lambda * mlm_loss

class _SingleAxisOperation(nn.Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        raise NotImplementedError

# Some general purpose convenience layers
# ---------------------------------------


class Expand(_SingleAxisOperation):
    def forward(self, x):
        return x.unsqueeze(self.axis)


class Squeeze(_SingleAxisOperation):
    def forward(self, x):
        return x.squeeze(self.axis)


class Permute(nn.Module):
    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        return x.permute(self.axes)


class Concatenate(_SingleAxisOperation):
    def forward(self, *x):
        if len(x) == 1 and isinstance(x[0], tuple):
            x = x[0]
        return torch.cat(x, dim=self.axis)


class IndexSelect(nn.Module):
    def __init__(self, indices):
        super().__init__()
        assert isinstance(indices, (int, list, tuple))
        if isinstance(indices, int):
            indices = [indices]
        self.indices = list()
        for i in indices:
            assert isinstance(i, int)
            self.indices.append(i)

    def forward(self, *x):
        if len(x) == 1 and isinstance(x[0], tuple):
            x = x[0]
        if len(self.indices) == 1:
            return x[self.indices[0]]
        return [x[i] for i in self.indices]


class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class ConvBlock2D(nn.Module):
    """
    Implements complete convolution block with order:
      - Convolution
      - dropout (spatial)
      - activation
      - batch-norm
      - (optional) residual reconnection
    """

    def __init__(self, in_filters, out_filters, kernel, stride=(1, 1), padding=0, dilation=1, groups=1, do_rate=0.5,
                 batch_norm=True, activation=nn.LeakyReLU, residual=False):
        super().__init__()
        self.kernel = kernel
        self.activation = activation()
        self.residual = residual

        self.conv = nn.Conv2d(in_filters, out_filters, kernel, stride=stride, padding=padding, dilation=dilation,
                           groups=groups, bias=not batch_norm)
        self.dropout = nn.Dropout2d(p=do_rate)
        self.batch_norm = nn.BatchNorm2d(out_filters)

    def forward(self, input, **kwargs):
        res = input
        input = self.conv(input, **kwargs)
        input = self.dropout(input)
        input = self.activation(input)
        input = self.batch_norm(input)
        return input + res if self.residual else input

# ---------------------------------------


# New layers
# ---------------------------------------


class DenseFilter(nn.Module):
    def __init__(self, in_features, growth_rate, filter_len=5, do=0.5, bottleneck=2, activation=nn.LeakyReLU, dim=-2):
        """
        This DenseNet-inspired filter block features in the TIDNet network from Kostas & Rudzicz 2020 (Thinker
        Invariance). 2D convolution is used, but with a kernel that only spans one of the dimensions. In TIDNet it is
        used to develop channel operations independently of temporal changes.

        Parameters
        ----------
        in_features
        growth_rate
        filter_len
        do
        bottleneck
        activation
        dim
        """
        super().__init__()
        dim = dim if dim > 0 else dim + 4
        if dim < 2 or dim > 3:
            raise ValueError('Only last two dimensions supported')
        kernel = (filter_len, 1) if dim == 2 else (1, filter_len)

        self.net = nn.Sequential(
            nn.BatchNorm2d(in_features),
            activation(),
            nn.Conv2d(in_features, bottleneck * growth_rate, 1),
            nn.BatchNorm2d(bottleneck * growth_rate),
            activation(),
            nn.Conv2d(bottleneck * growth_rate, growth_rate, kernel, padding=tuple((k // 2 for k in kernel))),
            nn.Dropout2d(do)
        )

    def forward(self, x):
        return torch.cat((x, self.net(x)), dim=1)


class DenseSpatialFilter(nn.Module):
    def __init__(self, channels, growth, depth, in_ch=1, bottleneck=4, dropout_rate=0.0, activation=nn.LeakyReLU,
                 collapse=True):
        """
        This extends the :any:`DenseFilter` to specifically operate in channel space and collapse this dimension
        over the course of `depth` layers.

        Parameters
        ----------
        channels
        growth
        depth
        in_ch
        bottleneck
        dropout_rate
        activation
        collapse
        """
        super().__init__()
        self.net = nn.Sequential(*[
            DenseFilter(in_ch + growth * d, growth, bottleneck=bottleneck, do=dropout_rate,
                        activation=activation) for d in range(depth)
        ])
        n_filters = in_ch + growth * depth
        self.collapse = collapse
        if collapse:
            self.channel_collapse = ConvBlock2D(n_filters, n_filters, (channels, 1), do_rate=0)

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(1).permute([0, 1, 3, 2])
        x = self.net(x)
        if self.collapse:
            return self.channel_collapse(x).squeeze(-2)
        return x


class SpatialFilter(nn.Module):
    def __init__(self, channels, filters, depth, in_ch=1, dropout_rate=0.0, activation=nn.LeakyReLU, batch_norm=True,
                 residual=False):
        super().__init__()
        kernels = [(channels // depth, 1) for _ in range(depth-1)]
        kernels += [(channels - sum(x[0] for x in kernels) + depth-1, 1)]
        self.filter = nn.Sequential(
            ConvBlock2D(in_ch, filters, kernels[0], do_rate=dropout_rate/depth, activation=activation,
                        batch_norm=batch_norm),
            *[ConvBlock2D(filters, filters, kernel, do_rate=dropout_rate/depth, activation=activation,
                          batch_norm=batch_norm)
              for kernel in kernels[1:]]
        )
        self.residual = nn.Conv1d(channels * in_ch, filters, 1) if residual else None

    def forward(self, x):
        res = x
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
        elif self.residual:
            res = res.contiguous().view(res.shape[0], -1, res.shape[3])
        x = self.filter(x).squeeze(-2)
        return x + self.residual(res) if self.residual else x


class TemporalFilter(nn.Module):

    def __init__(self, channels, filters, depth, temp_len, dropout=0., activation=nn.LeakyReLU, residual='netwise'):
        """
        This implements the dilated temporal-only spanning convolution from TIDNet.

        Parameters
        ----------
        channels
        filters
        depth
        temp_len
        dropout
        activation
        residual
        """
        super().__init__()
        temp_len = temp_len + 1 - temp_len % 2
        self.residual_style = str(residual)
        net = list()

        for i in range(depth):
            dil = depth - i
            conv = nn.utils.weight_norm(nn.Conv2d(channels if i == 0 else filters, filters, kernel_size=(1, temp_len),
                                      dilation=dil, padding=(0, dil * (temp_len - 1) // 2)))
            net.append(nn.Sequential(
                conv,
                activation(),
                nn.Dropout2d(dropout)
            ))
        if self.residual_style.lower() == 'netwise':
            self.net = nn.Sequential(*net)
            self.residual = nn.Conv2d(channels, filters, (1, 1))
        elif residual.lower() == 'dense':
            self.net = net

    def forward(self, x):
        if self.residual_style.lower() == 'netwise':
            return self.net(x) + self.residual(x)
        elif self.residual_style.lower() == 'dense':
            for l in self.net:
                x = torch.cat((x, l(x)), dim=1)
            return x



class _BENDREncoder(nn.Module):
    def __init__(self, in_features, encoder_h=256,):
        super().__init__()
        self.in_features = in_features
        self.encoder_h = encoder_h

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze


class ConvEncoderBENDR(_BENDREncoder):
    def __init__(self, in_features, encoder_h=256, enc_width=(3, 2, 2, 2, 2, 2),
                 dropout=0., projection_head=False, enc_downsample=(3, 2, 2, 2, 2, 2)):
        super().__init__(in_features, encoder_h)
        self.encoder_h = encoder_h
        if not isinstance(enc_width, (list, tuple)):
            enc_width = [enc_width]
        if not isinstance(enc_downsample, (list, tuple)):
            enc_downsample = [enc_downsample]
        assert len(enc_downsample) == len(enc_width)

        # Centerable convolutions make life simpler
        enc_width = [e if e % 2 else e+1 for e in enc_width]
        self._downsampling = enc_downsample
        self._width = enc_width

        self.encoder = nn.Sequential()
        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module("Encoder_{}".format(i), nn.Sequential(
                nn.Conv1d(in_features, encoder_h, width, stride=downsample, padding=width // 2),
                nn.Dropout2d(dropout),
                nn.GroupNorm(encoder_h // 2, encoder_h),
                nn.GELU(),
            ))
            in_features = encoder_h

        if projection_head:
            self.encoder.add_module("projection-1", nn.Sequential(
                nn.Conv1d(in_features, in_features, 1),
                nn.Dropout2d(dropout*2),
                nn.GroupNorm(in_features // 2, in_features),
                nn.GELU()
            ))

    def description(self, sfreq=None, sequence_len=None):
        widths = list(reversed(self._width))[1:]
        strides = list(reversed(self._downsampling))[1:]

        rf = self._width[-1]
        for w, s in zip(widths, strides):
            rf = rf if w == 1 else (rf - 1) * s + 2 * (w // 2)

        desc = "Receptive field: {} samples".format(rf)
        if sfreq is not None:
            desc += ", {:.2f} seconds".format(rf / sfreq)

        ds_factor = np.prod(self._downsampling)
        desc += " | Downsampled by {}".format(ds_factor)
        if sfreq is not None:
            desc += ", new sfreq: {:.2f} Hz".format(sfreq / ds_factor)
        desc += " | Overlap of {} samples".format(rf - ds_factor)
        if sequence_len is not None:
            desc += " | {} encoded samples/trial".format(sequence_len // ds_factor)
        return desc

    def downsampling_factor(self, samples):
        for factor in self._downsampling:
            samples = np.ceil(samples / factor)
        return samples

    def forward(self, x):
        return self.encoder(x)


# FIXME this is redundant with part of the contextualizer
class EncodingAugment(nn.Module):
    def __init__(self, in_features, mask_p_t=0.1, mask_p_c=0.01, mask_t_span=6, mask_c_span=64, dropout=0.1,
                 position_encoder=25):
        super().__init__()
        self.mask_replacement = torch.nn.Parameter(torch.zeros(in_features), requires_grad=True)
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        transformer_dim = 3 * in_features

        conv = nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
        nn.init.normal_(conv.weight, mean=0, std=2 / transformer_dim)
        nn.init.constant_(conv.bias, 0)
        conv = nn.utils.weight_norm(conv, dim=2)
        self.relative_position = nn.Sequential(conv, nn.GELU())

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, transformer_dim, 1),
        )

    def forward(self, x, mask_t=None, mask_c=None):
        bs, feat, seq = x.shape

        if self.training:
            if mask_t is None and self.p_t > 0 and self.mask_t_span > 0:
                mask_t = _make_mask((bs, seq), self.p_t, x.shape[-1], self.mask_t_span)
            if mask_c is None and self.p_c > 0 and self.mask_c_span > 0:
                mask_c = _make_mask((bs, feat), self.p_c, x.shape[1], self.mask_c_span)

        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        x = self.input_conditioning(x + self.relative_position(x))
        return x

    def init_from_contextualizer(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=False)
        for param in self.parameters():
            param.requires_grad = False
        print("Initialized mask embedding and position encoder from ", filename)


class _Hax(nn.Module):
    """T-fixup assumes self-attention norms are removed"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class BENDRContextualizer(nn.Module):

    def __init__(self, in_features, hidden_feedforward=3076, heads=8, layers=8, dropout=0.15, activation='gelu',
                 position_encoder=25, layer_drop=0.0, mask_p_t=0.1, mask_p_c=0.004, mask_t_span=6, mask_c_span=64,
                 start_token=-5, finetuning=False):
        super(BENDRContextualizer, self).__init__()

        self.dropout = dropout
        self.in_features = in_features
        self._transformer_dim = in_features * 3

        encoder = nn.TransformerEncoderLayer(d_model=in_features * 3, nhead=heads, dim_feedforward=hidden_feedforward,
                                             dropout=dropout, activation=activation)
        encoder.norm1 = _Hax()
        encoder.norm2 = _Hax()

        self.norm = nn.LayerNorm(self._transformer_dim)

        # self.norm_layers = nn.ModuleList([copy.deepcopy(norm) for _ in range(layers)])
        self.transformer_layers = nn.ModuleList([copy.deepcopy(encoder) for _ in range(layers)])
        self.layer_drop = layer_drop
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        self.start_token = start_token
        self.finetuning = finetuning

        # Initialize replacement vector with 0's
        self.mask_replacement = torch.nn.Parameter(torch.normal(0, in_features**(-0.5), size=(in_features,)),
                                                   requires_grad=True)

        self.position_encoder = position_encoder > 0
        if position_encoder:
            conv = nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
            nn.init.normal_(conv.weight, mean=0, std=2 / self._transformer_dim)
            nn.init.constant_(conv.bias, 0)
            conv = nn.utils.weight_norm(conv, dim=2)
            self.relative_position = nn.Sequential(conv, nn.GELU())

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, self._transformer_dim, 1),
            Permute([2, 0, 1]),
        )

        self.output_layer = nn.Conv1d(self._transformer_dim, in_features, 1)
        self.apply(self.init_bert_params)

    def init_bert_params(self, module):
        if isinstance(module, nn.Linear):
            # module.weight.data.normal_(mean=0.0, std=0.02)
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            # Tfixup
            module.weight.data = 0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data

        # if isinstance(module, nn.Conv1d):
        #     # std = np.sqrt((4 * (1.0 - self.dropout)) / (self.in_features * self.in_features))
        #     # module.weight.data.normal_(mean=0.0, std=std)
        #     nn.init.xavier_uniform_(module.weight.data)
        #     module.bias.data.zero_()

    def forward(self, x, mask_t=None, mask_c=None):
        bs, feat, seq = x.shape
        if self.training and self.finetuning:
            if mask_t is None and self.p_t > 0:
                mask_t = _make_mask((bs, seq), self.p_t, x.shape[-1], self.mask_t_span)
            if mask_c is None and self.p_c > 0:
                mask_c = _make_mask((bs, feat), self.p_c, x.shape[1], self.mask_c_span)

        # Multi-gpu workaround, wastes memory
        x = x.clone()

        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        if self.position_encoder:
            x = x + self.relative_position(x)
        x = self.input_conditioning(x)

        if self.start_token is not None:
            in_token = self.start_token * torch.ones((1, 1, 1), requires_grad=True).to(x.device).expand([-1, *x.shape[1:]])
            x = torch.cat([in_token, x], dim=0)

        for layer in self.transformer_layers:
            if not self.training or torch.rand(1) > self.layer_drop:
                x = layer(x)

        return self.output_layer(x.permute([1, 2, 0]))

    def freeze_features(self, unfreeze=False, finetuning=False):
        for param in self.parameters():
            param.requires_grad = unfreeze
        if self.finetuning or finetuning:
            self.mask_replacement.requires_grad = False

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

class DN3ConfigException(BaseException):
    """
    Exception to be triggered when DN3-configuration parsing fails.
    """
    pass

def _fif_raw_or_epoch(fname, preload=True):
    # See https://mne.tools/stable/generated/mne.read_epochs.html
    if str(fname).endswith('-epo.fif'):
        return mne.read_epochs(fname, preload=preload)
    else:
        return loader.read_raw_fif(fname, preload=preload)

_SUPPORTED_EXTENSIONS = {
    '.edf': loader.read_raw_edf,
    # FIXME: need to handle part fif files
    '.fif': _fif_raw_or_epoch,

    # TODO: add much more support, at least all of MNE-python
    '.bdf': loader.read_raw_bdf,
    '.gdf': loader.read_raw_gdf,
}

# These are hard-coded in MOABB, if you are having trouble with this, check if the "sign" has changed
SUPPORTED_DATASETS = {
    'BNCI2014001': mbd.BNCI2014001,
    'PhysionetMI': mbd.PhysionetMI,
    'Cho2017': mbd.Cho2017
}

class _DumbNamespace:
    def __init__(self, d: dict):
        self._d = d.copy()
        for k in d:
            if isinstance(d[k], dict):
                d[k] = _DumbNamespace(d[k])
            if isinstance(d[k], list):
                d[k] = [_DumbNamespace(d[k][i]) if isinstance(d[k][i], dict) else d[k][i] for i in range(len(d[k]))]
        self.__dict__.update(d)

    def keys(self):
        return list(self.__dict__.keys())

    def __getitem__(self, item):
        return self.__dict__[item]

    def as_dict(self):
        return self._d

def _adopt_auxiliaries(obj, remaining):
    def namespaceify(v):
        if isinstance(v, dict):
            return _DumbNamespace(v)
        elif isinstance(v, list):
            return [namespaceify(v[i]) for i in range(len(v))]
        else:
            return v

    obj.__dict__.update({k: namespaceify(v) for k, v in remaining.items()})

class MoabbDataset:

    def __init__(self, ds_name, data_location, **kwargs):
        try:
            self.ds = SUPPORTED_DATASETS[ds_name](**kwargs)
        except KeyError:
            raise DN3ConfigException("No support for MOABB dataset called {}".format(ds_name))
        self.path = data_location
        self.data_dict = None
        self.run_map = dict()

    def _get_ds_data(self):
        if self.data_dict is None:
            self.ds.download(path=str(self.path), update_path=True)
            self.data_dict = self.ds.get_data()

    def get_pseudo_mapping(self, exclusion_cb):
        self._get_ds_data()
        # self.run_map = {th: dict() for th in self.data_dict.keys()}
        # DN3 collapses sessions and runs
        mapping = dict()

        for th in self.data_dict.keys():
            for sess in self.data_dict[th].keys():
                for run in self.data_dict[th][sess].keys():
                    id = '-'.join((str(th), str(sess), str(run)))
                    self.run_map[id] = self.data_dict[th][sess][run]
                    if exclusion_cb(self.data_dict[th][sess][run].filenames[0], str(th), id):
                        continue
                    if th in mapping:
                        mapping[th].append(id)
                    else:
                        mapping[th] = [id]
        return mapping

    def get_raw(self, pseudo_path):
        return self.run_map[str(pseudo_path)]

class _Recording(DN3ataset, ABC):
    """
    Abstract base class for any supported recording
    """
    def __init__(self, info, session_id, person_id, tlen, ch_ind_picks=None):
        super().__init__()
        self.info = info
        self.picks = ch_ind_picks if ch_ind_picks is not None else list(range(len(info['chs'])))
        self._recording_channels = [(ch['ch_name'], int(ch['kind'])) for idx, ch in enumerate(info['chs'])
                                    if idx in self.picks]
        self._recording_sfreq = info['sfreq']
        self._recording_len = int(self._recording_sfreq * tlen)
        assert self._recording_sfreq is not None
        self.session_id = session_id
        self.person_id = person_id

    def get_all(self):
        all_recordings = [self[i] for i in range(len(self))]
        return [torch.stack(t) for t in zip(*all_recordings)]

    @property
    def sfreq(self):
        sfreq = self._recording_sfreq
        for xform in self._transforms:
            sfreq = xform.new_sfreq(sfreq)
        return sfreq

    @property
    def channels(self):
        channels = np.array(self._recording_channels)
        for xform in self._transforms:
            channels = xform.new_channels(channels)
        return channels

    @property
    def sequence_length(self):
        sequence_length = self._recording_len
        for xform in self._transforms:
            sequence_length = xform.new_sequence_length(sequence_length)
        return sequence_length


class RawTorchRecording(_Recording):
    """
    Interface for bridging mne Raw instances as PyTorch compatible "Dataset".

    Parameters
    ----------
    raw : mne.io.Raw
          Raw data, data does not need to be preloaded.
    tlen : float
          Length of recording specified in seconds.
    session_id : (int, str, optional)
          A unique (with respect to a thinker within an eventual dataset) identifier for the current recording
          session. If not specified, defaults to '0'.
    person_id : (int, str, optional)
          A unique (with respect to an eventual dataset) identifier for the particular person being recorded.
    stride : int
          The number of samples to skip between each starting offset of loaded samples.
    """

    def __init__(self, raw: mne.io.Raw, tlen, session_id=0, person_id=0, stride=1, ch_ind_picks=None, decimate=1,
                 bad_spans=None, **kwargs):

        """
        Interface for bridging mne Raw instances as PyTorch compatible "Dataset".

        Parameters
        ----------
        raw : mne.io.Raw
              Raw data, data does not need to be preloaded.
        tlen : float
              Length of each retrieved portion of the recording.
        session_id : (int, str, optional)
              A unique (with respect to a thinker within an eventual dataset) identifier for the current recording
              session. If not specified, defaults to '0'.
        person_id : (int, str, optional)
              A unique (with respect to an eventual dataset) identifier for the particular person being recorded.
        stride : int
              The number of samples to skip between each starting offset of loaded samples.
        ch_ind_picks : list[int]
                       A list of channel indices that have been selected for.
        decimate : int
                   The number of samples to move before taking the next sample, in other words take every decimate'th
                   sample.
        bad_spans: List[tuple], None
                   These are tuples of (start_seconds, end_seconds) of times that should be avoided. Any sequences that
                   would overlap with these sections will be excluded.
        """
        super().__init__(raw.info, session_id, person_id, tlen, ch_ind_picks)
        self.filename = raw.filenames[0]
        self.decimate = int(decimate)
        self._recording_sfreq /= self.decimate
        self._recording_len = int(tlen * self._recording_sfreq)
        self.stride = stride
        # Implement my own (rather than mne's) in-memory buffer when there are savings
        self._stride_load = self.decimate > 1 and raw.preload
        self.max = kwargs.get('max', None)
        self.min = kwargs.get('min', 0)
        bad_spans = list() if bad_spans is None else bad_spans
        self.__dict__.update(kwargs)

        self._decimated_sequence_starts = list(
            range(0, raw.n_times // self.decimate - self._recording_len, self.stride)
        )
        # TODO come back to this inefficient BS
        for start, stop in bad_spans:
            start = int(self._recording_sfreq * start)
            stop = int(stop * self._recording_sfreq)
            drop = list()
            for i, span_start in enumerate(self._decimated_sequence_starts):
                if start <= span_start < stop or start <= span_start + self._recording_len <= stop:
                    drop.append(span_start)
            for span_start in drop:
                self._decimated_sequence_starts.remove(span_start)

        # When the stride is greater than the sequence length, preload savings can be found by chopping the
        # sequence into subsequences of length: sequence length. Also, if decimating, can significantly reduce memory
        # requirements not otherwise addressed with the Raw object.
        if self._stride_load and len(self._decimated_sequence_starts) > 0:
            x = raw.get_data(self.picks)
            # pre-decimate this data for more preload savings (and for the stride factors to be valid)
            x = x[:, ::decimate]
            self._x = np.empty([x.shape[0], self._recording_len, len(self._decimated_sequence_starts)], dtype=x.dtype)
            for i, start in enumerate(self._decimated_sequence_starts):
                self._x[..., i] = x[:, start:start + self._recording_len]
        else:
            self._raw_workaround(raw)

    def _raw_workaround(self, raw):
        self.raw = raw

    def __getitem__(self, index):
        if index < 0:
            index += len(self)

        if self._stride_load:
            x = self._x[:, :, index]
        else:
            start = self._decimated_sequence_starts[index]
            x = self.raw.get_data(self.picks, start=start, stop=start + self._recording_len * self.decimate)
            if self.decimate > 1:
                x = x[:, ::self.decimate]

        scale = 1 if self.max is None else (x.max() - x.min()) / (self.max - self.min)
        if scale > 1 or np.isnan(scale):
            print('Warning: scale exeeding 1')

        x = torch.from_numpy(x).float()

        if torch.any(torch.isnan(x)):
            print("Nan found: raw {}, index {}".format(self.filename, index))
            print("Replacing with random values with same shape for now...")
            x = torch.rand_like(x)

        return self._execute_transforms(x)

    def __len__(self):
        return len(self._decimated_sequence_starts)

    def preprocess(self, preprocessor: Preprocessor, apply_transform=True):
        self.raw = preprocessor(recording=self)
        if apply_transform:
            self.add_transform(preprocessor.get_transform())

class RawOnTheFlyRecording(RawTorchRecording):

    def __init__(self, raw, tlen, file_loader, session_id=0, person_id=0, stride=1, ch_ind_picks=None,
                 decimate=1, **kwargs):
        """
        This provides a workaround for the normal raw recording pipeline so that files are not loaded in any way until
        they are needed. MNE's Raw object are too bloated for extremely large datasets, even without preloading.

        Parameters
        ----------
        raw
        tlen
        file_loader
        session_id
        person_id
        stride
        ch_ind_picks
        decimate
        kwargs
        """
        super().__init__(raw, tlen, session_id, person_id, stride, ch_ind_picks, decimate, **kwargs)
        self.file_loader = file_loader

    def _raw_workaround(self, raw):
        return

    @property
    def raw(self):
        with warnings.catch_warnings():
            # Ignore loading warnings that were already addressed during configuratron start-up
            warnings.simplefilter("ignore")
            return self.file_loader(self.filename)

    def preprocess(self, preprocessor, apply_transform=True):
        result = preprocessor(recording=self)
        if result is not None:
            raise DN3ConfigException("Modifying raw after preprocessing is not supported with on-the-fly")
        if apply_transform:
            self.add_transform(preprocessor.get_transform())

def make_epochs_from_raw(raw: mne.io.Raw, tmin, tlen, event_ids=None, baseline=None, decim=1, filter_bp=None,
                         drop_bad=False, use_annotations=False, chunk_duration=None):
    sfreq = raw.info['sfreq']
    if filter_bp is not None:
        if isinstance(filter_bp, (list, tuple)) and len(filter_bp) == 2:
            raw.load_data()
            raw.filter(filter_bp[0], filter_bp[1])
        else:
            print('Filter must be provided as a two-element list [low, high]')

    try:
        if use_annotations:
            events = mne.events_from_annotations(raw, event_id=event_ids, chunk_duration=chunk_duration)[0]
        else:
            events = mne.find_events(raw)
            events = events[[i for i in range(len(events)) if events[i, -1] in event_ids.keys()], :]
    except ValueError as e:
        raise DN3ConfigException(*e.args)

    return mne.Epochs(raw, events, tmin=tmin, tmax=tmin + tlen - 1 / sfreq, preload=True, decim=decim,
                      baseline=baseline, reject_by_annotation=drop_bad)

class EpochTorchRecording(_Recording):
    def __init__(self, epochs: mne.Epochs, session_id=0, person_id=0, force_label=None, cached=False,
                 ch_ind_picks=None, event_mapping=None, skip_epochs=None):
        """
        Wraps :any:`mne.Epochs` instances so that they conform to the :any:`Recording` API.

        Parameters
        ----------
        epochs
        session_id
        person_id
        force_label : bool, Optional
                      Whether to force the labels provided by the epoch instance. By default (False), will convert
                      output label (for N classes) into codes 0 -> N-1.
        cached
        ch_ind_picks
        event_mapping : dict, Optional
                        Mapping of human-readable names to numeric codes used by `epochs`.
        skip_epochs: List[int]
                    A list of epochs to skip
        """
        super().__init__(epochs.info, session_id, person_id, epochs.tmax - epochs.tmin + 1 / epochs.info['sfreq'],
                         ch_ind_picks)
        self.epochs = epochs
        # TODO scrap this cache option, it seems utterly redundant now
        self._cache = [None for _ in range(len(epochs.events))] if cached else None
        if event_mapping is None:
            # mne parses this for us
            event_mapping = epochs.event_id
        if force_label:
            self.epoch_codes_to_class_labels = event_mapping
        else:
            reverse_mapping = {v: k for k, v in event_mapping.items()}
            self.epoch_codes_to_class_labels = {v: i for i, v in enumerate(sorted(reverse_mapping.keys()))}
        skip_epochs = list() if skip_epochs is None else skip_epochs
        self._skip_map = [i for i in range(len(self.epochs.events)) if i not in skip_epochs]
        self._skip_map = dict(zip(range(len(self._skip_map)), self._skip_map))

    def __getitem__(self, index):
        index = self._skip_map[index]
        ep = self.epochs[index]

        if self._cache is None or self._cache[index] is None:
            # TODO Could have a speedup if not using ep, but items, but would need to drop bads?
            x = ep.get_data(picks=self.picks)
            if len(x.shape) != 3 or 0 in x.shape:
                print("I don't know why: {} index{}/{}".format(self.epochs.filename, index, len(self)))
                print(self.epochs.info['description'])
                print("Using trial {} in place for now...".format(index-1))
                return self.__getitem__(index - 1)
            x = torch.from_numpy(x.squeeze(0)).float()
            if self._cache is not None:
                self._cache[index] = x
        else:
            x = self._cache[index]

        y = torch.tensor(self.epoch_codes_to_class_labels[ep.events[0, -1]]).squeeze().long()

        return self._execute_transforms(x, y)

    def __len__(self):
        return len(self._skip_map)

    def preprocess(self, preprocessor: Preprocessor, apply_transform=True, **kwargs):
        processed = preprocessor(self, **kwargs)
        if processed is not None:
            self._cache = [processed[i] for i in range(processed.shape[0])]
        if apply_transform:
            self.add_transform(preprocessor.get_transform())

    def event_mapping(self):
        """
        Maps the labels returned by this to the events as recorded in the original annotations or stim channel.

        Returns
        -------
        mapping : dict
                  Keys are the class labels used by this object, values are the original event signifier.
        """
        return self.epoch_codes_to_class_labels

    def get_targets(self):
        return np.apply_along_axis(lambda x: self.epoch_codes_to_class_labels[x[0]], 1,
                                   self.epochs.events[list(self._skip_map.values()), -1, np.newaxis]).squeeze()

_EXTRA_CHANNELS = 5
_LEFT_NUMBERS = list(reversed(range(1, 9, 2)))
_RIGHT_NUMBERS = list(range(2, 10, 2))
DEEP_1010_CHS_LISTING = [
    # EEG
    "NZ",
    "FP1", "FPZ", "FP2",
    "AF7", "AF3", "AFZ", "AF4", "AF8",
    "F9", *["F{}".format(n) for n in _LEFT_NUMBERS], "FZ", *["F{}".format(n) for n in _RIGHT_NUMBERS], "F10",

    "FT9", "FT7", *["FC{}".format(n) for n in _LEFT_NUMBERS[1:]], "FCZ",
    *["FC{}".format(n) for n in _RIGHT_NUMBERS[:-1]], "FT8", "FT10",
                                                                                                                                  
    "T9", "T7", "T3",  *["C{}".format(n) for n in _LEFT_NUMBERS[1:]], "CZ",
    *["C{}".format(n) for n in _RIGHT_NUMBERS[:-1]], "T4", "T8", "T10",

    "TP9", "TP7", *["CP{}".format(n) for n in _LEFT_NUMBERS[1:]], "CPZ",
    *["CP{}".format(n) for n in _RIGHT_NUMBERS[:-1]], "TP8", "TP10",

    "P9", "P7", "T5",  *["P{}".format(n) for n in _LEFT_NUMBERS[1:]], "PZ",
    *["P{}".format(n) for n in _RIGHT_NUMBERS[:-1]],  "T6", "P8", "P10",

    "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1",  "OZ", "O2",
    "IZ",
    # EOG
    "VEOGL", "VEOGR", "HEOGL", "HEOGR",

    # Ear clip references
    "A1", "A2", "REF",
    # SCALING
    "SCALE",
    # Extra
    *["EX{}".format(n) for n in range(1, _EXTRA_CHANNELS+1)]
]

DEEP_1010_SCALE_CH = NamedInt('DN3_DEEP1010_SCALE_CH', 3000)
DEEP_1010_EXTRA_CH = NamedInt('DN3_DEEP1010_EXTRA_CH', 3001)
EEG_INDS = list(range(0, DEEP_1010_CHS_LISTING.index('VEOGL')))
EOG_INDS = [DEEP_1010_CHS_LISTING.index(ch) for ch in ["VEOGL", "VEOGR", "HEOGL", "HEOGR"]]
REF_INDS = [DEEP_1010_CHS_LISTING.index(ch) for ch in ["A1", "A2", "REF"]]
EXTRA_INDS = list(range(len(DEEP_1010_CHS_LISTING) - _EXTRA_CHANNELS, len(DEEP_1010_CHS_LISTING)))
_NUM_EEG_CHS = len(DEEP_1010_CHS_LISTING) - len(EOG_INDS) - len(REF_INDS) - len(EXTRA_INDS) - 1
SCALE_IND = -len(EXTRA_INDS) + len(DEEP_1010_CHS_LISTING)
DEEP_1010_CH_TYPES = ([FIFF.FIFFV_EEG_CH] * _NUM_EEG_CHS) + ([FIFF.FIFFV_EOG_CH] * len(EOG_INDS)) + \
                     ([FIFF.FIFFV_EEG_CH] * len(REF_INDS)) + [DEEP_1010_SCALE_CH] + \
                     ([DEEP_1010_EXTRA_CH] * _EXTRA_CHANNELS)

def _likely_eeg_channel(name):
    if name is not None:
        for ch in DEEP_1010_CHS_LISTING[:_NUM_EEG_CHS]:
            if ch in name.upper():
                return True
    return False

def _valid_character_heuristics(name, informative_characters):
    possible = ''.join(c for c in name.upper() if c in informative_characters).replace(' ', '')
    if possible == "":
        print("Could not use channel {}. Could not resolve its true label, rename first.".format(name))
        return None
    return possible

def _heuristic_eog_resolution(eog_channel_name):
    return _valid_character_heuristics(eog_channel_name, "VHEOGLR")

def _heuristic_ref_resolution(ref_channel_name: str):
    ref_channel_name = ref_channel_name.replace('EAR', '')
    ref_channel_name = ref_channel_name.replace('REF', '')
    if ref_channel_name.find('A1') != -1:
        return 'A1'
    elif ref_channel_name.find('A2') != -1:
        return 'A2'

    if ref_channel_name.find('L') != -1:
        return 'A1'
    elif ref_channel_name.find('R') != -1:
        return 'A2'
    return "REF"

def _heuristic_eeg_resolution(eeg_ch_name: str):
    eeg_ch_name = eeg_ch_name.upper()
    # remove some common garbage
    eeg_ch_name = eeg_ch_name.replace('EEG', '')
    eeg_ch_name = eeg_ch_name.replace('REF', '')
    informative_characters = set([c for name in DEEP_1010_CHS_LISTING[:_NUM_EEG_CHS] for c in name])
    return _valid_character_heuristics(eeg_ch_name, informative_characters)


def _heuristic_resolution(old_type_dict: OrderedDict):
    resolver = {'eeg': _heuristic_eeg_resolution, 'eog': _heuristic_eog_resolution, 'ref': _heuristic_ref_resolution,
                'extra': lambda x: x, None: lambda x: x}

    new_type_dict = OrderedDict()

    for old_name, ch_type in old_type_dict.items():
        if ch_type is None:
            new_type_dict[old_name] = None
            continue

        new_name = resolver[ch_type](old_name)
        if new_name is None:
            new_type_dict[old_name] = None
        else:
            while new_name in new_type_dict.keys():
                print('Deep1010 Heuristics resulted in duplicate entries for {}, incrementing name, but will be lost '
                      'in mapping'.format(new_name))
                new_name = new_name + '-COPY'
            new_type_dict[new_name] = old_type_dict[old_name]

    assert len(new_type_dict) == len(old_type_dict)
    return new_type_dict

def _check_num_and_get_types(type_dict: OrderedDict):
    type_lists = list()
    for ch_type, max_num in zip(('eog', 'ref'), (len(EOG_INDS), len(REF_INDS))):
        channels = [ch_name for ch_name, _type in type_dict.items() if _type == ch_type]

        for name in channels[max_num:]:
            print("Losing assumed {} channel {} because there are too many.".format(ch_type, name))
            type_dict[name] = None
        type_lists.append(channels[:max_num])
    return type_lists[0], type_lists[1]

def _deep_1010(map, names, eog, ear_ref, extra):

    for i, ch in enumerate(names):
        if ch not in eog and ch not in ear_ref and ch not in extra:
            try:
                map[i, DEEP_1010_CHS_LISTING.index(str(ch).upper())] = 1.0
            except ValueError:
                print("Warning: channel {} not found in standard layout. Skipping...".format(ch))
                continue

    # Normalize for when multiple values are mapped to single location
    summed = map.sum(axis=0)[np.newaxis, :]
    mapping = torch.from_numpy(np.divide(map, summed, out=np.zeros_like(map), where=summed != 0)).float()
    mapping.requires_grad_(False)
    return mapping

def map_named_channels_deep_1010(channel_names: list, EOG=None, ear_ref=None, extra_channels=None):
    """
    Maps channel names to the Deep1010 format, will automatically map EOG and extra channels if they have been
    named according to standard convention. Otherwise provide as keyword arguments.

    Parameters
    ----------
    channel_names : list
                   List of channel names from dataset
    EOG : list, str
         Must be a single channel name, or left and right EOG channels, optionally vertical L/R then horizontal
         L/R for four channels.
    ear_ref : Optional, str, list
               One or two channels to be used as references. If two, should be left and right in that order.
    extra_channels : list, None
                     Up to 6 extra channels to include. Currently not standardized, but could include ECG, respiration,
                     EMG, etc.

    Returns
    -------
    mapping : torch.Tensor
              Mapping matrix from previous channel sequence to Deep1010.
    """
    map = np.zeros((len(channel_names), len(DEEP_1010_CHS_LISTING)))

    if isinstance(EOG, str):
        EOG = [EOG] * 4
    elif len(EOG) == 1:
        EOG = EOG * 4
    elif EOG is None or len(EOG) == 0:
        EOG = []
    elif len(EOG) == 2:
        EOG = EOG * 2
    else:
        assert len(EOG) == 4
    for eog_map, eog_std in zip(EOG, EOG_INDS):
        try:
            map[channel_names.index(eog_map), eog_std] = 1.0
        except ValueError:
            raise ValueError("EOG channel {} not found in provided channels.".format(eog_map))

    if isinstance(ear_ref, str):
        ear_ref = [ear_ref] * 2
    elif ear_ref is None:
        ear_ref = []
    else:
        assert len(ear_ref) <= len(REF_INDS)
    for ref_map, ref_std in zip(ear_ref, REF_INDS):
        try:
            map[channel_names.index(ref_map), ref_std] = 1.0
        except ValueError:
            raise ValueError("Reference channel {} not found in provided channels.".format(ref_map))

    if isinstance(extra_channels, str):
        extra_channels = [extra_channels]
    elif extra_channels is None:
        extra_channels = []
    assert len(extra_channels) <= _EXTRA_CHANNELS
    for ch, place in zip(extra_channels, EXTRA_INDS):
        if ch is not None:
            map[channel_names.index(ch), place] = 1.0

    return _deep_1010(map, channel_names, EOG, ear_ref, extra_channels)

def map_dataset_channels_deep_1010(channels: np.ndarray, exclude_stim=True):
    """
    Maps channels as stored by a :any:`DN3ataset` to the Deep1010 format, will automatically map EOG and extra channels
    by type.

    Parameters
    ----------
    channels : np.ndarray
               Channels that remain a 1D sequence (they should not have been projected into 2 or 3D grids) of name and
               type. This means the array has 2 dimensions:
               ..math:: N_{channels} \by 2
               With the latter dimension containing name and type respectively, as is constructed by default in most
               cases.
    exclude_stim : bool
                   This option allows the stim channel to be added as an *extra* channel. The default (True) will not do
                   this, and it is very rare if ever where this would be needed.

    Warnings
    --------
    If for some reason the stim channel is labelled with a label from the `DEEP_1010_CHS_LISTING` it will be included
    in that location and result in labels bleeding into the observed data.

    Returns
    -------
    mapping : torch.Tensor
              Mapping matrix from previous channel sequence to Deep1010.
    """
    if len(channels.shape) != 2 or channels.shape[1] != 2:
        raise ValueError("Deep1010 Mapping: channels must be a 2 dimensional array with dim0 = num_channels, dim1 = 2."
                         " Got {}".format(channels.shape))
    channel_types = OrderedDict()

    # Use this for some semblance of order in the "extras"
    extra = [None for _ in range(_EXTRA_CHANNELS)]
    extra_idx = 0

    for name, ch_type in channels:
        # Annoyingly numpy converts them to strings...
        ch_type = int(ch_type)
        if ch_type == FIFF.FIFFV_EEG_CH and _likely_eeg_channel(name):
            channel_types[name] = 'eeg'
        elif ch_type == FIFF.FIFFV_EOG_CH or name in [DEEP_1010_CHS_LISTING[idx] for idx in EOG_INDS]:
            channel_types[name] = 'eog'
        elif ch_type == FIFF.FIFFV_STIM_CH:
            if exclude_stim:
                channel_types[name] = None
                continue
            # if stim, always set as last extra
            channel_types[name] = 'extra'
            extra[-1] = name
        elif 'REF' in name.upper() or 'A1' in name.upper() or 'A2' in name.upper() or 'EAR' in name.upper():
            channel_types[name] = 'ref'
        else:
            if extra_idx == _EXTRA_CHANNELS - 1 and not exclude_stim:
                print("Stim channel overwritten by {} in Deep1010 mapping.".format(name))
            elif extra_idx == _EXTRA_CHANNELS:
                print("No more room in extra channels for {}".format(name))
                continue
            channel_types[name] = 'extra'
            extra[extra_idx] = name
            extra_idx += 1

    revised_channel_types = _heuristic_resolution(channel_types)
    eog, ref = _check_num_and_get_types(revised_channel_types)

    return map_named_channels_deep_1010(list(revised_channel_types.keys()), eog, ref, extra)

def min_max_normalize(x: torch.Tensor, low=-1, high=1):
    if len(x.shape) == 2:
        xmin = x.min()
        xmax = x.max()
        if xmax - xmin == 0:
            x = 0
            return x
    elif len(x.shape) == 3:
        xmin = torch.min(torch.min(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        xmax = torch.max(torch.max(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        constant_trials = (xmax - xmin) == 0
        if torch.any(constant_trials):
            # If normalizing multiple trials, stabilize the normalization
            xmax[constant_trials] = xmax[constant_trials] + 1e-6

    x = (x - xmin) / (xmax - xmin)

    # Now all scaled 0 -> 1, remove 0.5 bias
    x -= 0.5
    # Adjust for low/high bias and scale up
    x += (high + low) / 2
    return (high - low) * x

class MappingDeep1010(InstanceTransform):
    """
    Maps various channel sets into the Deep10-10 scheme, and normalizes data between [-1, 1] with an additional scaling
    parameter to describe the relative scale of a trial with respect to the entire dataset.

    See https://doi.org/10.1101/2020.12.17.423197  for description.
    """
    def __init__(self, dataset, max_scale=None, return_mask=False):
        """
        Creates a Deep10-10 mapping for the provided dataset.

        Parameters
        ----------
        dataset : Dataset

        max_scale : float
                    If specified, the scale ind is filled with the relative scale of the trial with respect
                    to this, otherwise uses dataset.info.data_max - dataset.info.data_min.
        return_mask : bool
                      If `True` (`False` by default), an additional tensor is returned after this transform that
                      says which channels of the mapping are in fact in use.
        """
        super().__init__()
        self.mapping = map_dataset_channels_deep_1010(dataset.channels)
        if max_scale is not None:
            self.max_scale = max_scale
        elif dataset.info is None or dataset.info.data_max is None or dataset.info.data_min is None:
            print(f"Warning: Did not find data scale information for {dataset}")
            self.max_scale = None
            pass
        else:
            self.max_scale = dataset.info.data_max - dataset.info.data_min
        self.return_mask = return_mask

    @staticmethod
    def channel_listing():
        return DEEP_1010_CHS_LISTING

    def __call__(self, x):
        if self.max_scale is not None:
            scale = 2 * (torch.clamp_max((x.max() - x.min()) / self.max_scale, 1.0) - 0.5)
        else:
            scale = 0

        x = (x.transpose(1, 0) @ self.mapping).transpose(1, 0)

        for ch_type_inds in (EEG_INDS, EOG_INDS, REF_INDS, EXTRA_INDS):
            x[ch_type_inds, :] = min_max_normalize(x[ch_type_inds, :])

        used_channel_mask = self.mapping.sum(dim=0).bool()
        x[~used_channel_mask, :] = 0

        x[SCALE_IND, :] = scale

        if self.return_mask:
            return (x, used_channel_mask)
        else:
            return x

    def new_channels(self, old_channels: np.ndarray):
        channels = list()
        for row in range(self.mapping.shape[1]):
            active = self.mapping[:, row].nonzero().numpy()
            if len(active) > 0:
                channels.append("-".join([old_channels[i.item(), 0] for i in active]))
            else:
                channels.append(None)
        return np.array(list(zip(channels, DEEP_1010_CH_TYPES)))

def skip_inds_from_bad_spans(epochs: mne.Epochs, bad_spans: list):
    if bad_spans is None:
        return None

    start_times = epochs.events[:, 0] / epochs.info['sfreq']
    end_times = start_times + epochs.tmax - epochs.tmin

    skip_inds = list()
    for i, (start, end) in enumerate(zip(start_times, end_times)):
        for bad_start, bad_end in bad_spans:
            if bad_start <= start < bad_end or bad_start < end <= bad_end:
                skip_inds.append(i)
                break

    return skip_inds

class TemporalInterpolation(InstanceTransform):

    def __init__(self, desired_sequence_length, mode='nearest', new_sfreq=None):
        """
        This is in essence a DN3 wrapper for the pytorch function
        `interpolate() <https://pytorch.org/docs/stable/nn.functional.html>`_

        Currently only supports single dimensional samples (i.e. channels have not been projected into more dimensions)

        Warnings
        --------
        Using this function to downsample data below a suitable nyquist frequency given the low-pass filtering of the
        original data will cause dangerous aliasing artifacts that will heavily affect data quality to the point of it
        being mostly garbage.

        Parameters
        ----------
        desired_sequence_length: int
                                 The desired new sequence length of incoming data.
        mode: str
              The technique that will be used for upsampling data, by default 'nearest' interpolation. Other options
              are listed under pytorch's interpolate function.
        new_sfreq: float, None
                   If specified, registers the change in sampling frequency
        """
        super().__init__()
        self._new_sequence_length = desired_sequence_length
        self.mode = mode
        self._new_sfreq = new_sfreq

    def __call__(self, x):
        # squeeze and unsqueeze because these are done before batching
        if len(x.shape) == 2:
            return interpolate(x.unsqueeze(0), self._new_sequence_length, mode=self.mode).squeeze(0)
        # Supports batch dimension
        elif len(x.shape) == 3:
            return interpolate(x, self._new_sequence_length, mode=self.mode)
        else:
            raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")

    def new_sequence_length(self, old_sequence_length):
        return self._new_sequence_length

    def new_sfreq(self, old_sfreq):
        if self._new_sfreq is not None:
            return self._new_sfreq
        else:
            return old_sfreq

def unfurl(_set: set):
    _list = list(_set)
    for i in range(len(_list)):
        if not isinstance(_list[i], Iterable):
            _list[i] = [_list[i]]
    return tuple(x for z in _list for x in z)

def rand_split(dataset, frac=0.75):
    if frac >= 1:
        return dataset
    samples = len(dataset)
    return random_split(dataset, lengths=[round(x) for x in [samples*frac, samples*(1-frac)]])

class Thinker(DN3ataset, ConcatDataset):
    """
    Collects multiple recordings of the same person, intended to be of the same task, at different times or conditions.
    """

    def __init__(self, sessions, person_id="auto", return_session_id=False, return_trial_id=False,
                 propagate_kwargs=False):
        """
        Collects multiple recordings of the same person, intended to be of the same task, at different times or
        conditions.

        Parameters
        ----------
        sessions : Iterable, dict
                   Either a sequence of recordings, or a mapping of session_ids to recordings. If the former, the
                   recording's session_id is preserved. If the
        person_id : int, str
                    Label to be used for the thinker. If set to "auto" (default), will automatically pick the person_id
                    using the most common person_id in the recordings.
        return_session_id : bool
                           Whether to return (enumerated - see `Dataset`) session_ids with the data itself. Overridden
                           by `propagate_kwargs`, with key `session_id`
        propagate_kwargs : bool
                           If True, items are returned additional tensors generated by transforms, and session_id as
        """
        DN3ataset.__init__(self)
        if not isinstance(sessions, dict) and isinstance(sessions, Iterable):
            self.sessions = OrderedDict()
            for r in sessions:
                self.__add__(r)
        elif isinstance(sessions, dict):
            self.sessions = OrderedDict(sessions)
        else:
            raise TypeError("Recordings must be iterable or already processed dict.")
        if person_id == 'auto':
            ids = [sess.person_id for sess in self.sessions.values()]
            person_id = max(set(ids), key=ids.count)
        self.person_id = person_id

        for sess in self.sessions.values():
            sess.person_id = person_id

        self._reset_dataset()
        self.return_session_id = return_session_id
        self.return_trial_id = return_trial_id

    def _reset_dataset(self):
        for _id in self.sessions:
            self.sessions[_id].session_id = _id
        ConcatDataset.__init__(self, self.sessions.values())

    def __str__(self):
        return "Person {} - {} trials | {} transforms".format(self.person_id, len(self), len(self._transforms))

    @property
    def sfreq(self):
        sfreq = set(self.sessions[s].sfreq for s in self.sessions)
        if len(sfreq) > 1:
            print("Warning: Multiple sampling frequency values found. Over/re-sampling may be necessary.")
            return unfurl(sfreq)
        sfreq = sfreq.pop()
        for xform in self._transforms:
            sfreq = xform.new_sfreq(sfreq)
        return sfreq

    @property
    def channels(self):
        channels = [self.sessions[s].channels for s in self.sessions]
        if not same_channel_sets(channels):
            raise ValueError("Multiple channel sets found. A consistent mapping like Deep1010 is necessary to proceed.")
        channels = channels.pop()
        for xform in self._transforms:
            channels = xform.new_channels(channels)
        return channels

    @property
    def sequence_length(self):
        sequence_length = set(self.sessions[s].sequence_length for s in self.sessions)
        if len(sequence_length) > 1:
            print("Warning: Multiple sequence lengths found. A cropping transformation may be in order.")
            return unfurl(sequence_length)
        sequence_length = sequence_length.pop()
        for xform in self._transforms:
            sequence_length = xform.new_sequence_length(sequence_length)
        return sequence_length

    def __add__(self, sessions):
        assert isinstance(sessions, (_Recording, Thinker))
        if isinstance(sessions, Thinker):
            if sessions.person_id != self.person_id:
                print("Person IDs don't match: adding {} to {}. Assuming latter...")
            sessions = sessions.sessions

        if sessions.session_id in self.sessions.keys():
            self.sessions[sessions.session_id] += sessions
        else:
            self.sessions[sessions.session_id] = sessions

        self._reset_dataset()

    def pop_session(self, session_id, apply_thinker_transform=True):
        assert session_id in self.sessions.keys()
        sess = self.sessions.pop(session_id)
        if apply_thinker_transform:
            for xform in self._transforms:
                sess.add_transform(xform)
        self._reset_dataset()
        return sess

    def __getitem__(self, item, return_id=False):
        x = list(ConcatDataset.__getitem__(self, item))
        session_idx = bisect.bisect_right(self.cumulative_sizes, item)
        idx_offset = 2 if len(x) > 1 and x[1].dtype == torch.bool else 1
        if self.return_trial_id:
            trial_id = item if session_idx == 0 else item - self.cumulative_sizes[session_idx-1]
            x.insert(idx_offset, torch.tensor(trial_id).long())
        if self.return_session_id:
            x.insert(idx_offset, torch.tensor(session_idx).long())
        return self._execute_transforms(*x)

    def __len__(self):
        return ConcatDataset.__len__(self)

    def _make_like_me(self, sessions: Iterable):
        if not isinstance(sessions, dict):
            sessions = {s: self.sessions[s] for s in sessions}
        like_me = Thinker(sessions, self.person_id, self.return_session_id)
        for x in self._transforms:
            like_me.add_transform(x)
        return like_me

    def split(self, training_sess_ids=None, validation_sess_ids=None, testing_sess_ids=None, test_frac=0.25,
              validation_frac=0.25):
        """
        Split the thinker's data into training, validation and testing sets.

        Parameters
        ----------
        test_frac : float
                    Proportion of the total data to use for testing, this is overridden by `testing_sess_ids`.
        validation_frac : float
                          Proportion of the data remaining - after removing test proportion/sessions - to use as
                          validation data. Likewise, `validation_sess_ids` overrides this value.
        training_sess_ids : : (Iterable, None)
                            The session ids to be explicitly used for training.
        validation_sess_ids : (Iterable, None)
                            The session ids to be explicitly used for validation.
        testing_sess_ids : (Iterable, None)
                           The session ids to be explicitly used for testing.

        Returns
        -------
        training : DN3ataset
                   The training dataset
        validation : DN3ataset
                   The validation dataset
        testing : DN3ataset
                   The testing dataset
        """
        training_sess_ids = set(training_sess_ids) if training_sess_ids is not None else set()
        validation_sess_ids = set(validation_sess_ids) if validation_sess_ids is not None else set()
        testing_sess_ids = set(testing_sess_ids) if testing_sess_ids is not None else set()
        duplicated_ids = training_sess_ids.intersection(validation_sess_ids).intersection(testing_sess_ids)
        if len(duplicated_ids) > 0:
            print("Ids duplicated across train/val/test split: {}".format(duplicated_ids))
        use_sessions = self.sessions.copy()
        training, validating, testing = (
            self._make_like_me({s_id: use_sessions.pop(s_id) for s_id in ids}) if len(ids) else None
            for ids in (training_sess_ids, validation_sess_ids, testing_sess_ids)
        )
        if training is not None and validating is not None and testing is not None:
            if len(use_sessions) > 0:
                print("Warning: sessions specified do not span all sessions. Skipping {} sessions.".format(
                    len(use_sessions)))
                return training, validating, testing

        # Split up the rest if there is anything left
        if len(use_sessions) > 0:
            remainder = self._make_like_me(use_sessions.keys())
            if testing is None:
                assert test_frac is not None and 0 < test_frac < 1
                remainder, testing = rand_split(remainder, frac=test_frac)
            if validating is None:
                assert validation_frac is not None and 0 <= test_frac < 1
                if validation_frac > 0:
                    validating, remainder = rand_split(remainder, frac=validation_frac)

        training = remainder if training is None else training

        return training, validating, testing

    def preprocess(self, preprocessor: Preprocessor, apply_transform=True, sessions=None, **kwargs):
        """
        Applies a preprocessor to the dataset

        Parameters
        ----------
        preprocessor : Preprocessor
                       A preprocessor to be applied
        sessions : (None, Iterable)
                   If specified (default is None), the sessions to use for preprocessing calculation
        apply_transform : bool
                          Whether to apply the transform to this dataset (all sessions, not just those specified for
                          preprocessing) after preprocessing them. Exclusive application to select sessions can be
                          done using the return value and a separate call to `add_transform` with the same `sessions`
                          list.

        Returns
        ---------
        preprocessor : Preprocessor
                       The preprocessor after application to all relevant sessions
        """
        for sid, session in enumerate(self.sessions.values()):
            session.preprocess(preprocessor, session_id=sid, apply_transform=apply_transform, **kwargs)
        return preprocessor

    def clear_transforms(self, deep_clear=False):
        self._transforms = list()
        if deep_clear:
            for s in self.sessions.values():
                s.clear_transforms()

    def add_transform(self, transform, deep=False):
        if deep:
            for s in self.sessions.values():
                s.add_transform(transform)
        else:
            self._transforms.append(transform)

    def get_targets(self):
        """
        Collect all the targets (i.e. labels) that this Thinker's data is annotated with.

        Returns
        -------
        targets: np.ndarray
                 A numpy-formatted array of all the targets/label for this thinker.
        """
        targets = list()
        for sess in self.sessions:
            if hasattr(self.sessions[sess], 'get_targets'):
                targets.append(self.sessions[sess].get_targets())
        if len(targets) == 0:
            return None
        return np.concatenate(targets)

class DatasetInfo(object):
    """
    This objects contains non-critical meta-data that might need to be tracked for :py:`Dataset` objects. Generally
    not necessary to be constructed manually, these are created by the configuratron to automatically create transforms
    and/or other processes downstream.
    """
    def __init__(self, dataset_name, data_max=None, data_min=None, excluded_people=None, targets=None):
        self.__dict__.update(dict(dataset_name=dataset_name, data_max=data_max, data_min=data_min,
                                  excluded_people=excluded_people, targets=targets))

    def __str__(self):
        return "{} | {} targets | Excluding {}".format(self.dataset_name, self.targets, self.excluded_people)

class Dataset(DN3ataset, ConcatDataset):
    """
    Collects thinkers, each of which may collect multiple recording sessions of the same tasks, into a dataset with
    (largely) consistent:
      - hardware:
        - channel number/labels
        - sampling frequency
      - annotation paradigm:
        - consistent event types
    """
    def __init__(self, thinkers, dataset_id=None, task_id=None, return_trial_id=False, return_session_id=False,
                 return_person_id=False, return_dataset_id=False, return_task_id=False, dataset_info=None):
        """
        Collects recordings from multiple people, intended to be of the same task, at different times or
        conditions.
        Optionally, can specify whether to return person, session, dataset and task labels. Person and session ids will
        be converted to an enumerated set of integer ids, rather than those provided during creation of those datasets
        in order to make a minimal set of labels. e.g. if there are 3 thinkers, {A01, A02, and A05}, specifying
        `return_person_id` will return an additional tensor with 0 for A01, 1 for A02 and 2 for A05 respectively. To
        recover any original identifier, get_thinkers() returns a list of the original thinker ids such that the
        enumerated offset recovers the original identity. Extending the example above:
        ``self.get_thinkers()[1] == "A02"``

        .. warning:: The enumerated ids above are only ever used in the construction of model input tensors,
                     otherwise, anywhere where ids are required as API, the *human readable* version is uesd
                     (e.g. in our example above A02)

        Parameters
        ----------
        thinkers : Iterable, dict
                   Either a sequence of `Thinker`, or a mapping of person_id to `Thinker`. If the latter, id's are
                   overwritten by these id's.
        dataset_id : int
                     An identifier associated with data from the entire dataset. Unlike person and sessions, this should
                     simply be an integer for the sake of returning labels that can functionally be used for learning.
        task_id : int
                  An identifier associated with data from the entire dataset, and potentially others of the same task.
                  Like dataset_idm this should simply be an integer.
        return_person_id : bool
                           Whether to return (enumerated - see above) person_ids with the data itself.
        return_session_id : bool
                           Whether to return (enumerated - see above) session_ids with the data itself.
        return_dataset_id : bool
                           Whether to return the dataset_id with the data itself.
        return_task_id : bool
                           Whether to return the dataset_id with the data itself.
        return_trial_id: bool
                        Whether to return the id of the trial (within the session)
        dataset_info : DatasetInfo, Optional
                       Additional, non-critical data that helps specify additional features of the dataset.

        Notes
        -----------
        When getting items from a dataset, the id return order is returned most general to most specific, wrapped by
        the actual raw data and (optionally, if epoch-variety recordings) the label for the raw data, thus:
        raw_data, task_id, dataset_id, person_id, session_id, *label
        """
        super().__init__()
        self.info = dataset_info

        if not isinstance(thinkers, Iterable):
            raise ValueError("Provided thinkers must be in an iterable container, e.g. list, tuple, dicts")

        # Overwrite thinker ids with those provided as dict argument and sort by ids
        if not isinstance(thinkers, dict):
            thinkers = {t.person_id: t for t in thinkers}

        self.thinkers = OrderedDict()
        for t in sorted(thinkers.keys()):
            self.__add__(thinkers[t], person_id=t, return_session_id=return_session_id, return_trial_id=return_trial_id)
        self._reset_dataset()

        self.dataset_id = torch.tensor(dataset_id).long() if dataset_id is not None else None
        self.task_id = torch.tensor(task_id).long() if task_id is not None else None
        self.update_id_returns(return_trial_id, return_session_id, return_person_id, return_dataset_id, return_task_id)

    def update_id_returns(self, trial=None, session=None, person=None, task=None, dataset=None):
        """
        Updates which ids are to be returned by the dataset. If any argument is `None` it preserves the previous value.

        Parameters
        ----------
        trial : None, bool
                  Whether to return trial ids.
        session : None, bool
                  Whether to return session ids.
        person : None, bool
                 Whether to return person ids.
        task    : None, bool
                  Whether to return task ids.
        dataset : None, bool
                 Whether to return dataset ids.
        """
        self.return_trial_id = self.return_trial_id if trial is None else trial
        self.return_session_id = self.return_session_id if session is None else session
        self.return_person_id = self.return_person_id if person is None else person
        self.return_dataset_id = self.return_dataset_id if dataset is None else dataset
        self.return_task_id = self.return_task_id if task is None else task
        def set_ids_for_thinkers(th_id, thinker: Thinker):
            thinker.return_trial_id = self.return_trial_id
            thinker.return_session_id = self.return_session_id
        self._apply(set_ids_for_thinkers)

    def _reset_dataset(self):
        for p_id in self.thinkers:
            self.thinkers[p_id].person_id = p_id
            for s_id in self.thinkers[p_id].sessions:
                self.thinkers[p_id].sessions[s_id].session_id = s_id
                self.thinkers[p_id].sessions[s_id].person_id = p_id
        ConcatDataset.__init__(self, self.thinkers.values())

    def _apply(self, lam_fn):
        for th_id, thinker in self.thinkers.items():
            lam_fn(th_id, thinker)

    def __str__(self):
        ds_name = "Dataset-{}".format(self.dataset_id) if self.info is None else self.info.dataset_name
        return ">> {} | DSID: {} | {} people | {} trials | {} channels | {} samples/trial | {}Hz | {} transforms".\
            format(ds_name, self.dataset_id, len(self.get_thinkers()), len(self), len(self.channels),
                   self.sequence_length, self.sfreq, len(self._transforms))

    def __add__(self, thinker, person_id=None, return_session_id=None, return_trial_id=None):
        assert isinstance(thinker, Thinker)
        return_session_id = self.return_session_id if return_session_id is None else return_session_id
        return_trial_id = self.return_trial_id if return_trial_id is None else return_trial_id
        thinker.return_session_id = return_session_id
        thinker.return_trial_id = return_trial_id
        if person_id is not None:
            thinker.person_id = person_id

        if thinker.person_id in self.thinkers.keys():
            print("Warning. Person {} already in dataset... Merging sessions.".format(thinker.person_id))
            self.thinkers[thinker.person_id] += thinker
        else:
            self.thinkers[thinker.person_id] = thinker
        self._reset_dataset()
        return self

    def pop_thinker(self, person_id, apply_ds_transforms=False):
        assert person_id in self.get_thinkers()
        thinker = self.thinkers.pop(person_id)
        if apply_ds_transforms:
            for xform in self._transforms:
                thinker.add_transform(xform)
        self._reset_dataset()
        return thinker

    def __getitem__(self, item):
        person_id = bisect.bisect_right(self.cumulative_sizes, item)
        person = self.thinkers[self.get_thinkers()[person_id]]
        if person_id == 0:
            sample_idx = item
        else:
            sample_idx = item - self.cumulative_sizes[person_id - 1]
        x = list(person.__getitem__(sample_idx))

        if self._safe_mode:
            for i in range(len(x)):
                if torch.any(torch.isnan(x[i])):
                    raise DN3atasetNanFound("Nan found at tensor offset {}. "
                                            "Loading data from person {} and sample {}".format(i, person, sample_idx))

        # Skip deep1010 mask
        idx_offset = 2 if len(x) > 1 and x[1].dtype == torch.bool else 1
        if self.return_person_id:
            x.insert(idx_offset, torch.tensor(person_id).long())

        if self.return_dataset_id:
            x.insert(idx_offset, self.dataset_id)

        if self.return_task_id:
            x.insert(idx_offset, self.task_id)

        if self._safe_mode:
            try:
                return self._execute_transforms(*x)
            except DN3atasetNanFound as e:
                raise DN3atasetNanFound(
                    "Nan found after transform | {} | from person {} and sample {}".format(e.args, person, sample_idx))

        return self._execute_transforms(*x)

    def safe_mode(self, mode=True):
        """
        This allows switching *safe_mode* on or off. When safe_mode is on, if data is ever NaN, it is captured
        before being returned and a report is generated.

        Parameters
        ----------
        mode : bool
             The status of whether in safe mode or not.
        """
        self._safe_mode = mode

    def preprocess(self, preprocessor: Preprocessor, apply_transform=True, thinkers=None):
        """
        Applies a preprocessor to the dataset

        Parameters
        ----------
        preprocessor : Preprocessor
                       A preprocessor to be applied
        thinkers : (None, Iterable)
                   If specified (default is None), the thinkers to use for preprocessing calculation
        apply_transform : bool
                          Whether to apply the transform to this dataset (all thinkers, not just those specified for
                          preprocessing) after preprocessing them. Exclusive application to specific thinkers can be
                          done using the return value and a separate call to `add_transform` with the same `thinkers`
                          list.

        Returns
        ---------
        preprocessor : Preprocessor
                       The preprocessor after application to all relevant thinkers
        """
        for thid, thinker in enumerate(self.thinkers.values()):
            thinker.preprocess(preprocessor, thinker_id=thid, apply_transform=apply_transform)
        return preprocessor

    @property
    def sfreq(self):
        sfreq = set(self.thinkers[t].sfreq for t in self.thinkers)
        if len(sfreq) > 1:
            print("Warning: Multiple sampling frequency values found. Over/re-sampling may be necessary.")
            return unfurl(sfreq)
        sfreq = sfreq.pop()
        for xform in self._transforms:
            sfreq = xform.new_sfreq(sfreq)
        return sfreq

    @property
    def channels(self):
        channels = [self.thinkers[t].channels for t in self.thinkers]
        if not same_channel_sets(channels):
            raise ValueError("Multiple channel sets found. A consistent mapping like Deep1010 is necessary to proceed.")
        channels = channels.pop()
        for xform in self._transforms:
            channels = xform.new_channels(channels)
        return channels

    @property
    def sequence_length(self):
        sequence_length = set(self.thinkers[t].sequence_length for t in self.thinkers)
        if len(sequence_length) > 1:
            print("Warning: Multiple sequence lengths found. A cropping transformation may be in order.")
            return unfurl(sequence_length)
        sequence_length = sequence_length.pop()
        for xform in self._transforms:
            sequence_length = xform.new_sequence_length(sequence_length)
        return sequence_length

    def get_thinkers(self):
        """
        Accumulates a consistently ordered list of all the thinkers in the dataset. It is this order that any automatic
        segmenting through :py:meth:`loso()` and :py:meth:`lmso()` will be done.

        Returns
        -------
        thinker_names : list
        """
        return list(self.thinkers.keys())

    def get_sessions(self):
        """
        Accumulates all the sessions from each thinker in the dataset in a nested dictionary.

        Returns
        -------
        session_dict: dict
                      Keys are the thinkers of :py:meth:`get_thinkers()`, values are each another dictionary that maps
                      session ids to :any:`_Recording`
        """
        return {th: self.thinkers[th].sessions.copy() for th in self.thinkers}

    def __len__(self):
        self._reset_dataset()
        return self.cumulative_sizes[-1]

    def _make_like_me(self, people: list):
        if len(people) == 1:
            like_me = self.thinkers[people[0]].clone()
        else:
            dataset_id = self.dataset_id.item() if self.dataset_id is not None else None
            task_id = self.task_id.item() if self.task_id is not None else None

            like_me = Dataset({p: self.thinkers[p] for p in people}, dataset_id, task_id,
                              return_person_id=self.return_person_id, return_session_id=self.return_session_id,
                              return_dataset_id=self.return_dataset_id, return_task_id=self.return_task_id,
                              return_trial_id=self.return_trial_id, dataset_info=self.info)
        for x in self._transforms:
            like_me.add_transform(x)
        return like_me

    def _generate_splits(self, validation, testing):
        for val, test in zip(validation, testing):
            training = list(self.thinkers.keys())
            for v in val:
                training.remove(v)
            for t in test:
                training.remove(t)

            training = self._make_like_me(training)

            validating = self._make_like_me(val)
            _val_set = set(validating.get_thinkers()) if len(val) > 1 else {validating.person_id}

            testing = self._make_like_me(test)
            _test_set = set(testing.get_thinkers()) if len(test) > 1 else {testing.person_id}

            if len(_val_set.intersection(_test_set)) > 0:
                raise ValueError("Validation and test overlap with ids: {}".format(_val_set.intersection(_test_set)))

            print('Training:   {}'.format(training))
            print('Validation: {}'.format(validating))
            print('Test:       {}'.format(testing))

            yield training, validating, testing

    def loso(self, validation_person_id=None, test_person_id=None):
        """
        This *generates* a "Leave-one-subject-out" (LOSO) split. Tests each person one-by-one, and validates on the
        previous (the first is validated with the last).

        Parameters
        ----------
        validation_person_id : (int, str, list, optional)
                               If specified, and corresponds to one of the person_ids in this dataset, the loso cross
                               validation will consistently generate this thinker as `validation`. If *list*, must
                               be the same length as `test_person_id`, say a length N. If so, will yield N
                               each in sequence, and use remainder for test.
        test_person_id : (int, str, list, optional)
                         Same as `validation_person_id`, but for testing. However, testing may be a list when
                         validation is a single value. Thus if testing is N ids, will yield N values, with a consistent
                         single validation person. If a single id (int or str), and `validation_person_id` is not also
                         a single id, will ignore `validation_person_id` and loop through all others that are not the
                         `test_person_id`.

        Yields
        -------
        training : Dataset
                   Another dataset that represents the training set
        validation : Thinker
                     The validation thinker
        test : Thinker
               The test thinker
        """
        if isinstance(test_person_id, (str, int)) and isinstance(validation_person_id, (str, int)):
            yield from self._generate_splits([[validation_person_id]], [[test_person_id]])
            return
        elif isinstance(test_person_id, str):
            yield from self._generate_splits([[v] for v in self.get_thinkers() if v != test_person_id],
                                             [[test_person_id] for _ in range(len(self.get_thinkers()) - 1)])
            return

        # Testing is now either a sequence or nothing. Should loop over everyone (unless validation is a single id)
        if test_person_id is None and isinstance(validation_person_id, (str, int)):
            test_person_id = [t for t in self.get_thinkers() if t != validation_person_id]
            validation_person_id = [validation_person_id for _ in range(len(test_person_id))]
        elif test_person_id is None:
            test_person_id = [t for t in self.get_thinkers()]

        if validation_person_id is None:
            validation_person_id = [test_person_id[i - 1] for i in range(len(test_person_id))]

        if not isinstance(test_person_id, list) or len(test_person_id) != len(validation_person_id):
            raise ValueError("Test ids must be same length iterable as validation ids.")

        yield from self._generate_splits([[v] for v in validation_person_id], [[t] for t in test_person_id])

    def lmso(self, folds=10, test_splits=None, validation_splits=None):
        """
        This *generates* a "Leave-multiple-subject-out" (LMSO) split. In other words X-fold cross-validation, with
        boundaries enforced at thinkers (each person's data is not split into different folds).

        Parameters
        ----------
        folds : int
                If this is specified and `splits` is None, will split the subjects into this many folds, and then use
                each fold as a test set in turn (and the previous fold - starting with the last - as validation).
        test_splits : list, tuple
                This should be a list of tuples/lists of either:
                  - The ids of the consistent test set. In which case, folds must be specified, or validation_splits
                    is a nested list that .
                  - Two sub lists, first testing, second validation ids

        Yields
        -------
        training : Dataset
                   Another dataset that represents the training set
        validation : Dataset
                     The validation people as a dataset
        test : Thinker
               The test people as a dataset
        """

        def is_nested(split: list):
            should_be_nested = isinstance(split[0], (list, tuple))
            for x in split[1:]:
                if (should_be_nested and not isinstance(x, (list, tuple))) or (isinstance(x, (list, tuple))
                                                                               and not should_be_nested):
                        raise ValueError("Can't mix list/tuple and other elements when specifying ids.")
            if not should_be_nested and folds is None:
                raise ValueError("Can't infer folds from non-nested list. Specify folds, or nest ids")
            return should_be_nested

        def calculate_from_remainder(known_split):
            _folds = len(known_split) if is_nested(list(known_split)) else folds
            if folds is None:
                print("Inferred {} folds from test split.".format(_folds))
            remainder = list(set(self.get_thinkers()).difference(known_split))
            return [list(x) for x in np.array_split(remainder, _folds)], [known_split for _ in range(_folds)]

        if test_splits is None and validation_splits is None:
            if folds is None:
                raise ValueError("Must specify <folds> if not specifying ids.")
            folds = [list(x) for x in np.array_split(self.get_thinkers(), folds)]
            test_splits, validation_splits = zip(*[(folds[i], folds[i-1]) for i in range(len(folds))])
        elif validation_splits is None:
            validation_splits, test_splits = calculate_from_remainder(test_splits)
        elif test_splits is None:
            test_splits, validation_splits = calculate_from_remainder(validation_splits)

        yield from self._generate_splits(validation_splits, test_splits)

    def add_transform(self, transform, deep=False):
        if deep:
            for t in self.thinkers.values():
                t.add_transform(transform, deep=deep)
        else:
            self._transforms.append(transform)

    def clear_transforms(self, deep_clear=False):
        self._transforms = list()
        if deep_clear:
            for t in self.thinkers.values():
                t.clear_transforms(deep_clear=deep_clear)

    def get_targets(self):
        """
        Collect all the targets (i.e. labels) that this Thinker's data is annotated with.

        Returns
        -------
        targets: np.ndarray
                 A numpy-formatted array of all the targets/label for this thinker.
        """
        targets = list()
        for tid in self.thinkers:
            if hasattr(self.thinkers[tid], 'get_targets'):
                targets.append(self.thinkers[tid].get_targets())
        if len(targets) == 0:
            return None
        try:
            return np.concatenate(targets)
        # Catch exceptions due to inability to concatenate real targets.
        except ValueError:
            return None

    def dump_dataset(self, toplevel, compressed=True, apply_transforms=True, summary_file='dataset-dump.npz',
                     chunksize=100):
        """
        Dumps the dataset to the directory specified by toplevel, with a single file per index.

        Parameters
        ----------
        toplevel : str
                 The toplevel location to dump the dataset to. This folder (and path) will be created if it does not
                 exist.
        apply_transforms: bool
                 Whether to apply the transforms while preparing the data to be saved.
        """
        if apply_transforms is False:
            raise NotImplementedError

        toplevel = Path(toplevel)
        toplevel.mkdir(exist_ok=True, parents=True)

        thinkers = self.thinkers.copy()
        inds = 0
        for k in self.thinkers.keys():
            thinkers[k] = np.arange(inds, inds+len(thinkers[k]))
            inds += len(thinkers[k])

        np.savez_compressed(toplevel / summary_file, version='0.0.1', sfreq=self.sfreq, channels=self.channels,
                            sequence_length=self.sequence_length, chunksize=chunksize, name=self.info.dataset_name,
                            thinkers=thinkers, real_length=len(self))

        for i in tqdm.trange(round(len(self) / chunksize), desc="Saving", unit='files'):
            fp = toplevel / str(i)
            accumulated = list()
            for j in range(min(chunksize, len(self) - i*chunksize)):
                accumulated.append(self[i*chunksize + j])
            accumulated = [torch.stack(z) for z in zip(*accumulated)]
            if compressed:
                np.savez_compressed(fp, *[t.numpy() for t in accumulated])
            else:
                np.savez(fp, *[t.numpy() for t in accumulated])

class DN3atasetException(BaseException):
    """
    Exception to be triggered when DN3-dataset-specific issues arise.
    """
    pass

class DumpedDataset(DN3ataset):

    def __init__(self, toplevel, cache_all=False, summary_file='dataset-dump.npz', info=None, cache_chunk_factor=0.1):
        super(DumpedDataset, self).__init__()
        self.toplevel = Path(toplevel)
        assert self.toplevel.exists()
        summary_file = self.toplevel / summary_file
        self.info = info
        self._summary = np.load(summary_file, allow_pickle=True)
        self.thinkers = self._summary['thinkers'].flat[0]
        self._chunksize = self._summary['chunksize']
        self._num_per_cache = int(cache_chunk_factor * self._chunksize)
        self._len = self._summary['real_length']
        assert summary_file.exists()
        self.filenames = sorted([f for f in self.toplevel.iterdir() if f.name != summary_file.name],
                                key=lambda f: int(f.stem))

        self.cache = np.empty((self._len, len(self.channels), self.sequence_length)) if cache_all else None
        self.aux_cache = [None for _ in range(self._len)] if cache_all else None

    def __str__(self):
        ds_name = self.info.dataset_name if self.info is not None else "Dumped"
        return ">> {} | {} people | {} trials | {} channels | {} samples/trial | {}Hz | {} transforms | ". \
               format("ds_name", len(self.thinkers), len(self), len(self.channels),
                      self.sequence_length, self.sfreq, len(self._transforms))

    def __len__(self):
        return self._len

    @property
    def sfreq(self):
        return self._summary['sfreq']

    @property
    def channels(self):
        return self._summary['channels']

    @property
    def sequence_length(self):
        return self._summary['sequence_length']

    def get_thinkers(self):
        return list(self.thinkers.keys())

    def preprocess(self, preprocessor: Preprocessor, apply_transform=True):
        raise DN3atasetException("Can't preprocess dumped dataset. Load from original files to do this.")

    def __getitem__(self, index):
        if self.aux_cache is not None and self.aux_cache[index] is not None:
            print(f"Hit: chunk {index // self._chunksize}, id: {id(self.cache[index // self._chunksize])}")
            data = [self.cache[index], *self.aux_cache[index]]

        else:
            idx = index // self._chunksize
            offset = index % self._chunksize

            data = np.load(self.filenames[idx], allow_pickle=True)
            if self.aux_cache is not None and self.aux_cache[index] is None:
                # Put all loaded indexes into the cache
                # for i in set([offset] + np.random.choice(range(self._chunksize), self._num_per_cache, replace=False)):
                #     self.cache[int(idx * self._chunksize) + i] = [torch.from_numpy(data[f])[i] for f in data.files]
                for i in [offset]:
                    self.cache[index] = torch.from_numpy(data['arr_0'])
                    self.aux_cache[index] = [torch.from_numpy(data[f])[i] for f in data.files[1:]]
                data = [self.cache[index], *self.aux_cache[index]]
            else:
                data = [torch.from_numpy(data[f])[offset] for f in data.files]

        return self._execute_transforms(*data)

def stringify_channel_mapping(original_names: list, mapping: np.ndarray):
    result = ''
    heuristically_mapped = list()

    def match_old_new_idx(old_idx, new_idx_set: list):
        new_names = [DEEP_1010_CHS_LISTING[i] for i in np.nonzero(mapping[old_idx, :])[0] if i in new_idx_set]
        return ','.join(new_names)

    for inds, label in zip([list(range(0, _NUM_EEG_CHS)), EOG_INDS, REF_INDS, EXTRA_INDS],
                           ['EEG', 'EOG', 'REF', 'EXTRA']):
        result += "{} (original(new)): ".format(label)
        for idx, name in enumerate(original_names):
            news = match_old_new_idx(idx, inds)
            if len(news) > 0:
                result += '{}({}) '.format(name, news)
                if news != name.upper():
                    heuristically_mapped.append('{}({}) '.format(name, news))
        result += '\n'

    result += 'Heuristically Assigned: ' + ' '.join(heuristically_mapped)

    return result

class DatasetConfig:
    """
    Parses dataset entries in DN3 config
    """
    def __init__(self, name: str, config: dict, adopt_auxiliaries=True, ext_handlers=None, deep1010=None,
                 samples=None, sfreq=None, preload=False, return_trial_ids=False, relative_directory=None):
        """
        Parses dataset entries in DN3 config

        Parameters
        ----------
        name : str
               The name of the dataset specified in the config. Will be replaced if the optional `name` field is present
               in the config.
        config : dict
                The configuration entry for the dataset
        ext_handlers : dict, optional
                       If specified, should be a dictionary that maps file extensions (with dot e.g. `.edf`) to a
                       callable that returns a `raw` instance given a string formatted path to a file.
        adopt_auxiliaries : bool
                            Adopt additional configuration entries as object variables.
        deep1010 : None, dict
                   If `None` (default) will not use the Deep1010 to map channels. If a dict, will add this transform
                   to each recording, with keyword arguments from the dict.
        samples: int, None
                 Experiment level sample length, superceded by dataset-specific configuration
        sfreq: float, None
               Experiment level sampling frequency to be adhered to, this will be enforced if not None, ignoring
               decimate.
        preload: bool
                 Whether to preload recordings when creating datasets from the configuration. Can also be specified with
                 `preload` configuratron entry.
        return_trial_ids: bool
                 Whether to construct recordings that return trial ids.
        relative_directory: Path
                 Path to reference *toplevel* configuration entry to (if not an absolute path)

        """
        self._original_config = dict(config).copy()

        # Optional args set, these help define which are required, so they come first
        def get_pop(key, default=None):
            config.setdefault(key, default)
            return config.pop(key)

        # Epoching relevant options
        # self.tlen = get_pop('tlen')
        self.filename_format = get_pop('filename_format')
        if self.filename_format is not None and not fnmatch(self.filename_format, '*{subject*}*'):
            raise DN3ConfigException("Name format must at least include {subject}!")
        self.annotation_format = get_pop('annotation_format')
        self.tmin = get_pop('tmin')
        self._create_raw_recordings = self.tmin is None
        self.picks = get_pop('picks')
        if self.picks is not None and not isinstance(self.picks, list):
            raise DN3ConfigException("Specifying picks must be done as a list. Not {}.".format(self.picks))
        self.decimate = get_pop('decimate', 1)
        self.baseline = get_pop('baseline')
        if self.baseline is not None:
            self.baseline = tuple(self.baseline)
        self.bandpass = get_pop('bandpass')
        self.drop_bad = get_pop('drop_bad', False)
        self.events = get_pop('events')
        if self.events is not None:
            if not isinstance(self.events, (dict, list)):
                self.events = {0: self.events}
            elif isinstance(self.events, list):
                self.events = dict(zip(self.events, range(len(self.events))))
            self.events = OrderedDict(self.events)
        self.force_label = get_pop('force_label', False)
        self.chunk_duration = get_pop('chunk_duration')
        self.rename_channels = get_pop('rename_channels', dict())
        if not isinstance(self.rename_channels, dict):
            raise DN3ConfigException("Renamed channels must map new values to old values.")
        self.exclude_channels = get_pop('exclude_channels', list())
        if not isinstance(self.exclude_channels, list):
            raise DN3ConfigException("Excluded channels must be in a list.")

        # other options
        self.data_max = get_pop('data_max')
        self.data_min = get_pop('data_min')
        self.name = get_pop('name', name)
        self.dataset_id = get_pop('dataset_id')
        self.preload = get_pop('preload', preload)
        self.dumped = get_pop('pre-dumped')
        self.hpf = get_pop('hpf', None)
        self.lpf = get_pop('lpf', None)
        self.filter_data = self.hpf is not None or self.lpf is not None
        if self.filter_data:
            self.preload = True
        self.stride = get_pop('stride', 1)
        self.extensions = get_pop('file_extensions', list(_SUPPORTED_EXTENSIONS.keys()))
        self.exclude_people = get_pop('exclude_people', list())
        self.exclude_sessions = get_pop('exclude_sessions', list())
        self.exclude = get_pop('exclude', dict())
        self.deep1010 = deep1010
        if self.deep1010 is not None and (self.data_min is None or self.data_max is None):
            print("Warning: Can't add scale index with dataset that is missing info.")
        self._different_deep1010s = list()
        self._targets = get_pop('targets', None)
        self._unique_events = set()
        self.return_trial_ids = return_trial_ids
        self.from_moabb = get_pop('moabb')

        self._samples = get_pop('samples', samples)
        self._sfreq = sfreq
        if sfreq is not None and self.decimate > 1:
            print("{}: No need to specify decimate ({}) when sfreq is set ({})".format(self.name, self.decimate, sfreq))
            self.decimate = 1

        # Funky stuff
        self._on_the_fly = get_pop('load_onthefly', False)

        # Required args
        # TODO refactor a bit
        try:
            self.toplevel = get_pop('toplevel')
            if self.toplevel is None:
                if self.from_moabb is None:
                    raise KeyError()
                else:
                    # TODO resolve the use of MOABB `get_dataset_path()` confusion with "signs" vs. name of dataset
                    self.toplevel = mne.get_config('MNE_DATA', default='~/mne_data')
            self.toplevel = self._determine_path(self.toplevel, relative_directory)
            self.toplevel = Path(self.toplevel).expanduser()
            self.tlen = config.pop('tlen') if self._samples is None else None
        except KeyError as e:
            raise DN3ConfigException("Could not find required value: {}".format(e.args[0]))
        if not self.toplevel.exists():
            raise DN3ConfigException("The toplevel {} for dataset {} does not exists".format(self.toplevel, self.name))

        # The rest
        if adopt_auxiliaries and len(config) > 0:
            print("Adding additional configuration entries: {}".format(config.keys()))
            _adopt_auxiliaries(self, config)

        self._extension_handlers = _SUPPORTED_EXTENSIONS.copy()
        if ext_handlers is not None:
            for ext in ext_handlers:
                self.add_extension_handler(ext, ext_handlers[ext])

        self._excluded_people = list()

        # Callbacks and custom loaders
        self._custom_thinker_loader = None
        self._thinker_callback = None
        self._custom_raw_loader = None
        self._session_callback = None

        # Extensions
        if self.from_moabb is not None:
            try:
                self.from_moabb = MoabbDataset(self.from_moabb.pop('name'), self.toplevel.resolve(), **self.from_moabb)
            except KeyError:
                raise DN3ConfigException("MOABB configuration is incorrect. Make sure to use 'name' under MOABB to "
                                         "specify a compatible dataset.")
            self._custom_raw_loader = self.from_moabb.get_raw

    _PICK_TYPES = ['meg', 'eeg', 'stim', 'eog', 'ecg', 'emg', 'ref_meg', 'misc', 'resp', 'chpi', 'exci', 'ias', 'syst',
                   'seeg', 'dipole', 'gof', 'bio', 'ecog', 'fnirs', 'csd', ]

    @staticmethod
    def _picks_as_types(picks):
        if picks is None:
            return False
        for pick in picks:
            if not isinstance(pick, str) or pick.lower() not in DatasetConfig._PICK_TYPES:
                return False
        return True

    @staticmethod
    def _determine_path(toplevel, relative_directory=None):
        if relative_directory is None or str(toplevel)[0] in '/~.':
            return toplevel
        return str((Path(relative_directory) / toplevel).expanduser())

    def add_extension_handler(self, extension: str, handler):
        """
        Provide callable code to create a raw instance from sessions with certain file extensions. This is useful for
        handling of custom file formats, while preserving a consistent experiment framework.

        Parameters
        ----------
        extension : str
                   An extension that includes the '.', e.g. '.csv'
        handler : callable
                  Callback with signature f(path_to_file: str) -> mne.io.Raw, mne.io.Epochs

        """
        assert callable(handler)
        self._extension_handlers[extension] = handler

    def scan_toplevel(self):
        """
        Scan the provided toplevel for all files that may belong to the dataset.

        Returns
        -------
        files: list
               A listing of all the candidate filepaths (before excluding those that match exclusion criteria).
        """
        files = list()
        pbar = tqdm.tqdm(self.extensions,
                         desc="Scanning {}. If there are a lot of files, this may take a while...".format(
                             self.toplevel))
        for extension in pbar:
            pbar.set_postfix(dict(extension=extension))
            files += self.toplevel.glob("**/*{}".format(extension))
        return files

    def _is_narrowly_excluded(self, person_name, session_name):
        if person_name in self.exclude.keys():
            if self.exclude[person_name] is None:
                self._excluded_people.append(person_name)
                return True
            assert isinstance(self.exclude[person_name], dict)
            if session_name in self.exclude[person_name].keys() and self.exclude[person_name][session_name] is None:
                return True
        return False

    def is_excluded(self, f: Path, person_name, session_name):
        if self._is_narrowly_excluded(person_name, session_name):
            return True

        if True in [fnmatch(person_name, pattern) for pattern in self.exclude_people]:
            self._excluded_people.append(person_name)
            return True

        session_exclusion_patterns = self.exclude_sessions.copy()
        if self.annotation_format is not None:
            # Some hacks over here, but it will do
            patt = self.annotation_format.format(subject='**', session='*')
            patt = patt.replace('**', '*')
            patt = patt.replace('**', '*')
            session_exclusion_patterns.append(patt)
        for exclusion_pattern in session_exclusion_patterns:
            for version in (f.stem, f.name):
                if fnmatch(version, exclusion_pattern):
                    return True
        return False

    def _get_person_name(self, f: Path):
        if self.filename_format is None:
            person = f.parent.name
        else:
            person = search(self.filename_format, str(f))
            if person is None:
                raise DN3ConfigException("Could not find person in {} using {}.".format(f.name, self.filename_format))
            person = person['subject']
        return person

    def _get_session_name(self, f: Path):
        if self.filename_format is not None and fnmatch(self.filename_format, "*{session*}*"):
            sess_name = search(self.filename_format, str(f))['session']
        else:
            sess_name = f.name
        return sess_name

    def auto_mapping(self, files=None, reset_exclusions=True):
        """
        Generates a mapping of sessions and people of the dataset, assuming files are stored in the structure:
        `toplevel`/(*optional - <version>)/<person-id>/<session-id>.{ext}

        Parameters
        -------
        files : list
                Optional list of files (convertible to `Path` objects, e.g. relative or absolute strings) to be used.
                If not provided, will use `scan_toplevel()`.

        Returns
        -------
        mapping : dict
                  The keys are of all the people in the dataset, and each value another similar mapping to that person's
                  sessions.
        """
        if reset_exclusions:
            self._excluded_people = list()

        files = self.scan_toplevel() if files is None else files
        mapping = dict()
        for sess_file in files:
            sess_file = Path(sess_file)
            try:
                person_name = self._get_person_name(sess_file)
                session_name = self._get_session_name(sess_file)
            except DN3ConfigException:
                continue

            if self.is_excluded(sess_file, person_name, session_name):
                continue

            if person_name in mapping:
                mapping[person_name].append(str(sess_file))
            else:
                mapping[person_name] = [str(sess_file)]

        return mapping

    def _add_deep1010(self, ch_names: list, deep1010map: np.ndarray, unused):
        for i, (old_names, old_map, unused, count) in enumerate(self._different_deep1010s):
            if np.all(deep1010map == old_map):
                self._different_deep1010s[i] = (old_names, old_map, unused, count+1)
                return
        self._different_deep1010s.append((ch_names, deep1010map, unused, 1))

    def add_custom_raw_loader(self, custom_loader):
        """
        This is used to provide a custom implementation of taking a filename, and returning a :any:`mne.io.Raw()`
        instance. If properly constructed, all further configuratron options, such as resampling, epoching, filtering
        etc. should occur automatically.

        This is used to load unconventional files, e.g. '.mat' files from matlab, or custom '.npy' arrays, etc.

        Notes
        -----
        Consider using :any:`mne.io.Raw.add_events()` to integrate otherwise difficult (for the configuratron) to better
        specify events for each recording.

        Parameters
        ----------
        custom_loader: callable
                       A function that expects a single :any:`pathlib.Path()` instance as argument and returns an
                       instance of :any:`mne.io.Raw()`. To gracefully ignore problematic sessions, raise
                       :any:`DN3ConfigException` within.

        """
        assert callable(custom_loader)
        self._custom_raw_loader = custom_loader

    def add_progress_callbacks(self, session_callback=None, thinker_callback=None):
        """
        Add callbacks to be invoked on successful loading of session and/or thinker. Optionally, these can modify the
        respective loaded instances.

        Parameters
        ----------
        session_callback:
                          A function that expects a single session argument and can modify the (or return an
                          alternative) session.

        thinker_callback:
                          The same as for session, but with Thinker instances.

        """
        self._session_callback = session_callback
        self._thinker_callback = thinker_callback

    def _load_raw(self, path: Path):
        if self._custom_raw_loader is not None:
            return self._custom_raw_loader(path)
        if path.suffix in self._extension_handlers:
            return self._extension_handlers[path.suffix](str(path), preload=self.preload)
        print("Handler for file {} with extension {} not found.".format(str(path), path.suffix))
        for ext in path.suffixes:
            if ext in self._extension_handlers:
                print("Trying {} instead...".format(ext))
                return self._extension_handlers[ext](str(path), preload=self.preload)

        raise DN3ConfigException("No supported/provided loader found for {}".format(str(path)))

    @staticmethod
    def _prepare_session(raw, tlen, decimate, desired_sfreq, desired_samples, picks, exclude_channels, rename_channels,
                         hpf, lpf):
        if hpf is not None or lpf is not None:
            raw = raw.filter(hpf, lpf)

        lowpass = raw.info.get('lowpass', None)
        raw_sfreq = raw.info['sfreq']
        new_sfreq = raw_sfreq / decimate if desired_sfreq is None else desired_sfreq

        # Don't allow violation of Nyquist criterion if sfreq is being changed
        if lowpass is not None and (new_sfreq < 2 * lowpass) and new_sfreq != raw_sfreq:
            raise DN3ConfigException("Could not create raw for {}. With lowpass filter {}, sampling frequency {} and "
                                     "new sfreq {}. This is going to have bad aliasing!".format(raw.filenames[0],
                                                                                                raw.info['lowpass'],
                                                                                                raw.info['sfreq'],
                                                                                                new_sfreq))

        # Leverage decimation first to match desired sfreq (saves memory)
        if desired_sfreq is not None:
            while (raw_sfreq // (decimate + 1)) >= new_sfreq:
                decimate += 1

        # Pick types
        picks = pick_types(raw.info, **{t: t in picks for t in DatasetConfig._PICK_TYPES}) \
            if DatasetConfig._picks_as_types(picks) else list(range(len(raw.ch_names)))

        # Exclude channel index by pattern match
        picks = ([idx for idx in picks if True not in [fnmatch(raw.ch_names[idx], pattern)
                                                       for pattern in exclude_channels]])

        # Rename channels
        renaming_map = dict()
        for new_ch, pattern in rename_channels.items():
            for old_ch in [raw.ch_names[idx] for idx in picks if fnmatch(raw.ch_names[idx], pattern)]:
                renaming_map[old_ch] = new_ch
        try:
            raw = raw.rename_channels(renaming_map)
        except ValueError as e:
            print("Error renaming channels from session: ", raw.filenames[0])
            print("Failed to rename ", raw.ch_names, " using ", renaming_map)
            raise DN3ConfigException("Skipping channel name issue.")

        tlen = desired_samples / new_sfreq if tlen is None else tlen

        return raw, tlen, picks, new_sfreq

    def _construct_session_from_config(self, session, sess_id, thinker_id):
        bad_spans = None
        if thinker_id in self.exclude.keys():
            if sess_id in self.exclude[thinker_id].keys():
                bad_spans = self.exclude[thinker_id][sess_id]
                if bad_spans is None:
                    raise DN3ConfigException("Skipping {} - {}".format(thinker_id, sess_id))

        def load_and_prepare(sess):
            if not isinstance(sess, Path):
                sess = Path(sess)
            r = self._load_raw(sess)
            return (sess, *self._prepare_session(r, self.tlen, self.decimate, self._sfreq, self._samples, self.picks,
                                                self.exclude_channels, self.rename_channels, self.hpf, self.lpf))
        sess, raw, tlen, picks, new_sfreq = load_and_prepare(session)

        # Fixme - deprecate the decimate option in favour of specifying desired sfreq's
        if self._create_raw_recordings:
            if self._on_the_fly:
                recording = RawOnTheFlyRecording(raw, tlen, lambda s: load_and_prepare(s)[1], stride=self.stride,
                                                 decimate=self.decimate, ch_ind_picks=picks, bad_spans=bad_spans)
            else:
                recording = RawTorchRecording(raw, tlen, stride=self.stride, decimate=self.decimate, ch_ind_picks=picks,
                                              bad_spans=bad_spans)
        else:
            use_annotations = self.events is not None and True in [isinstance(x, str) for x in self.events.keys()]
            if not isinstance(raw, (mne.Epochs, mne.epochs.EpochsFIF)):  # Annoying other epochs type
                if use_annotations and self.annotation_format is not None:
                    patt = self.annotation_format.format(subject=thinker_id, session=sess_id)
                    ann = [str(f) for f in session.parent.glob(patt)]
                    if len(ann) > 0:
                        if len(ann) > 1:
                            print("More than one annotation found for {}. Falling back to {}".format(patt, ann[0]))
                        raw.set_annotations(read_annotations(ann[0]))
                epochs = make_epochs_from_raw(raw, self.tmin, tlen, event_ids=self.events, baseline=self.baseline,
                                              decim=self.decimate, filter_bp=self.bandpass, drop_bad=self.drop_bad,
                                              use_annotations=use_annotations, chunk_duration=self.chunk_duration)
            else:
                epochs = raw
            event_map = {v: v for v in self.events.values()} if use_annotations else self.events

            self._unique_events = self._unique_events.union(set(np.unique(epochs.events[:, -1])))
            recording = EpochTorchRecording(epochs, ch_ind_picks=picks, event_mapping=event_map,
                                            force_label=self.force_label,
                                            skip_epochs=skip_inds_from_bad_spans(epochs, bad_spans))

        if len(recording) == 0:
            raise DN3ConfigException("The recording at {} has no viable training data with the configuration options "
                                     "provided. Consider excluding this file or changing parameters.".format(str(session
                                                                                                                 )))

        if self.deep1010 is not None:
            # FIXME dataset not fully formed, but we can hack together something for now
            _dum = _DumbNamespace(dict(channels=recording.channels, info=dict(data_max=self.data_max,
                                                                              data_min=self.data_min)))
            xform = MappingDeep1010(_dum, **self.deep1010)
            recording.add_transform(xform)
            self._add_deep1010([raw.ch_names[i] for i in picks], xform.mapping.numpy(),
                               [raw.ch_names[i] for i in range(len(raw.ch_names)) if i not in picks])

        if recording.sfreq != new_sfreq:
            new_sequence_len = int(tlen * new_sfreq) if self._samples is None else self._samples
            recording.add_transform(TemporalInterpolation(new_sequence_len, new_sfreq=new_sfreq))

        return recording

    def add_custom_thinker_loader(self, thinker_loader):
        """
        Add custom code to load a specific thinker from a set of session files.

        Warnings
        ----------
        For all intents and purposes, this circumvents most of the configuratron, and results in it being mostly
        a tool for organizing dataset files. Most of the options are not leveraged and must be implemented by the
        custom loader. Please open an issue if you'd like to develop this option further!

        Parameters
        ----------
        thinker_loader:
                        A function that takes a dict argument that consists of the session-ids that map to filenames
                        (str) of all the detected session for the given thinker and a second argument for the detected
                        name of the person. The function should return a single instance of type :any:`Thinker`.
                        To gracefully ignore the person, raise a :any:`DN3ConfigException`

        """
        self._custom_thinker_loader = thinker_loader

    def _construct_thinker_from_config(self, thinker: list, thinker_id):
        sessions = {self._get_session_name(Path(s)): s for s in thinker}
        if self._custom_thinker_loader is not None:
            thinker = self._custom_thinker_loader(sessions, thinker_id)
        else:
            sessions = dict()
            for sess in thinker:
                sess = Path(sess)
                sess_name = self._get_session_name(sess)
                try:
                    new_session = self._construct_session_from_config(sess, sess_name, thinker_id)
                    after_cb = None if self._session_callback is None else self._session_callback(new_session)
                    sessions[sess_name] = new_session if after_cb is None else after_cb
                except DN3ConfigException as e:
                    tqdm.tqdm.write("Skipping {}. Exception: {}.".format(sess_name, e.args))
            if len(sessions) == 0:
                raise DN3ConfigException
            thinker = Thinker(sessions)

        if self.deep1010 is not None:
            # Quick check for if Deep1010 was already added to sessions
            skip = False
            for s in thinker.sessions.values():
                if skip:
                    break
                for x in s._transforms:
                    if isinstance(x, MappingDeep1010):
                        skip = True
                        break
            if not skip:
                # FIXME dataset not fully formed, but we can hack together something for now
                og_channels = list(thinker.channels[:, 0])
                _dum = _DumbNamespace(dict(channels=thinker.channels, info=dict(data_max=self.data_max,
                                                                                            data_min=self.data_min)))
                xform = MappingDeep1010(_dum, **self.deep1010)
                thinker.add_transform(xform)
                self._add_deep1010(og_channels, xform.mapping.numpy(), [])

        if self._sfreq is not None and thinker.sfreq != self._sfreq:
            new_sequence_len = int(thinker.sequence_length * self._sfreq / thinker.sfreq) if self._samples is None \
                else self._samples
            thinker.add_transform(TemporalInterpolation(new_sequence_len, new_sfreq=self._sfreq))

        return thinker

    def auto_construct_dataset(self, mapping=None, **dsargs):
        """
        This creates a dataset using the config values. If tlen and tmin are specified in the config, creates epoched
        dataset, otherwise Raw.

        Parameters
        ----------
        mapping : dict, optional
                A dict specifying a list of sessions (as paths to files) for each person_id in the dataset. e.g.
                {
                  person_1: [sess_1.edf, ...],
                  person_2: [sess_1.edf],
                  ...
                }
                If not specified, will use `auto_mapping()` to generate.
        dsargs :
                Any additional arguments to feed for the creation of the dataset. i.e. keyword arguments to `Dataset`'s
                constructor (which id's to return). If `dataset_info` is provided here, it will override what was
                inferrable from the configuration file.

        Returns
        -------
        dataset : Dataset
                An instance of :any:`Dataset`, constructed according to mapping.
        """
        if self.dumped is not None:
            path = Path(self.dumped)
            if path.exists():
                tqdm.tqdm.write("Found pre-dumped dataset directory at {}".format(self.dumped))
                info = DatasetInfo(self.name, self.data_max, self.data_min, self._excluded_people,
                                   targets=self._targets if self._targets is not None else len(self._unique_events))
                dataset = DumpedDataset(path, info=info)
                tqdm.tqdm.write(str(dataset))
                return dataset
            else:
                tqdm.tqdm.write("Could not load pre-dumped data, falling back to original data...")

        if self.from_moabb:
            print("Creating dataset using MOABB...")
            mapping = self.from_moabb.get_pseudo_mapping(exclusion_cb=self.is_excluded)
            print("Converting MOABB format to DN3")
        if mapping is None:
            return self.auto_construct_dataset(self.auto_mapping(), **dsargs)

        file_types = "Raw" if self._create_raw_recordings else "Epoched"
        if self.preload:
            file_types = "Preloaded " + file_types
        print("Creating dataset of {} {} recordings from {} people.".format(sum(len(mapping[p]) for p in mapping),
                                                                            file_types, len(mapping)))
        description = "Loading {}".format(self.name)
        thinkers = dict()
        for t in tqdm.tqdm(mapping, desc=description, unit='person'):
            try:
                new_thinker = self._construct_thinker_from_config(mapping[t], t)
                after_cb = None if self._thinker_callback is None else self._thinker_callback(new_thinker)
                thinkers[t] = new_thinker if after_cb is None else after_cb
            except DN3ConfigException:
                tqdm.tqdm.write("None of the sessions for {} were usable. Skipping...".format(t))

        info = DatasetInfo(self.name, self.data_max, self.data_min, self._excluded_people,
                           targets=self._targets if self._targets is not None else len(self._unique_events))
        dsargs.setdefault('dataset_info', info)
        dsargs.setdefault('dataset_id', self.dataset_id)
        dsargs.setdefault('return_trial_id', self.return_trial_ids)
        dataset = Dataset(thinkers, **dsargs)
        print(dataset)
        if self.deep1010 is not None:
            print("Constructed {} channel maps".format(len(self._different_deep1010s)))
            for names, deep_mapping, unused, count in self._different_deep1010s:
                print('=' * 20)
                print("Used by {} recordings:".format(count))
                print(stringify_channel_mapping(names, deep_mapping))
                print('-'*20)
                print("Excluded {}".format(unused))
                print('=' * 20)
        #     dataset.add_transform(MappingDeep1010(dataset))
        return dataset

class To1020(InstanceTransform):

    EEG_20_div = [
               'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'T5', 'P3', 'PZ', 'P4', 'T6',
                'O1', 'O2'
    ]

    def __init__(self, only_trial_data=True, include_scale_ch=True, include_ref_chs=False):
        """
        Transforms incoming Deep1010 data into exclusively the more limited 1020 channel set.
        """
        super(To1020, self).__init__(only_trial_data=only_trial_data)
        self._inds_20_div = [DEEP_1010_CHS_LISTING.index(ch) for ch in self.EEG_20_div]
        if include_ref_chs:
            self._inds_20_div.append([DEEP_1010_CHS_LISTING.index(ch) for ch in ['A1', 'A2']])
        if include_scale_ch:
            self._inds_20_div.append(SCALE_IND)

    def new_channels(self, old_channels):
        return old_channels[self._inds_20_div]

    def __call__(self, *x):
        x = list(x)
        for i in range(len(x)):
            # Assume every tensor that has deep1010 length should be modified
            if len(x[i].shape) > 0 and x[i].shape[0] == len(DEEP_1010_CHS_LISTING):
                x[i] = x[i][self._inds_20_div, ...]
        return x


class ExperimentConfig:
    """
    Parses DN3 configuration files. Checking the DN3 token for listed datasets.
    """
    def __init__(self, config_filename: str, adopt_auxiliaries=True):
        """
        Parses DN3 configuration files. Checking the DN3 token for listed datasets.

        Parameters
        ----------
        config_filename : str
                          String for path to yaml formatted configuration file
        adopt_auxiliaries : bool
                             For any additional tokens aside from DN3 and specified datasets, integrate them into this
                             object for later use. Defaults to True. This will propagate for the detected datasets.
        """
        with open(config_filename, 'r') as fio:
            self._original_config = yaml.load(fio, Loader=yaml.FullLoader)
        working_config = self._original_config.copy()

        if 'Configuratron' not in working_config.keys():
            raise DN3ConfigException("Toplevel `Configuratron` not found in: {}".format(config_filename))
        if 'datasets' not in working_config.keys():
            raise DN3ConfigException("`datasets` not found in {}".format([k.lower() for k in
                                                                          working_config.keys()]))

        self.experiment = working_config.pop('Configuratron')

        ds_entries = working_config.pop('datasets')
        ds_entries = dict(zip(range(len(ds_entries)), ds_entries)) if isinstance(ds_entries, list) else ds_entries
        usable_datasets = list(ds_entries.keys())

        if self.experiment is None:
            self._make_deep1010 = dict()
            self.global_samples = None
            self.global_sfreq = None
            return_trial_ids = False
            preload = False
            relative_directory = None
        else:
            # If not None, will be used
            self._make_deep1010 = self.experiment.get('deep1010', dict())
            if isinstance(self._make_deep1010, bool):
                self._make_deep1010 = dict() if self._make_deep1010 else None
            self.global_samples = self.experiment.get('samples', None)
            self.global_sfreq = self.experiment.get('sfreq', None)
            usable_datasets = self.experiment.get('use_only', usable_datasets)
            preload = self.experiment.get('preload', False)
            return_trial_ids = self.experiment.get('trial_ids', False)
            relative_directory = self.experiment.get('relative_directory', None)

        self.datasets = dict()
        # for i, name in enumerate(usable_datasets):
        #     if name in ds_entries.keys():
        #         self.datasets[name] = DatasetConfig(name, ds_entries[name], deep1010=self._make_deep1010,
        #                                             samples=self.global_samples, sfreq=self.global_sfreq,
        #                                             preload=preload, return_trial_ids=return_trial_ids,
        #                                             relative_directory=relative_directory)
        #     else:
        #         raise DN3ConfigException("Could not find {} in datasets".format(name))

        # print("Configuratron found {} datasets.".format(len(self.datasets), "s" if len(self.datasets) > 0 else ""))

        # if adopt_auxiliaries:
        #     _adopt_auxiliaries(self, working_config)
        for name in usable_datasets:
            if name in ds_entries.keys():
                self.datasets[name] = {}
                for subject, sessions in ds_entries[name].items():
                    self.datasets[name][subject] = {}
                    for session, recordings in sessions.items():
                        self.datasets[name][subject][session] = DatasetConfig(
                            name, recordings, deep1010=self._make_deep1010,
                            samples=self.global_samples, sfreq=self.global_sfreq,
                            preload=preload, return_trial_ids=return_trial_ids,
                            relative_directory=relative_directory
                        )
            else:
                raise DN3ConfigException("Could not find {} in datasets".format(name))

        print("Configuratron found {} datasets.".format(len(self.datasets)))

        if adopt_auxiliaries:
            self._adopt_auxiliaries(working_config)

class RandomTemporalCrop(BatchTransform):

    def __init__(self, max_crop_frac=0.25, temporal_axis=1):
        """
        Uniformly crops the time-dimensions of a batch.

        Parameters
        ----------
        max_crop_frac: float
                       The is the maximum fraction to crop off of the trial.
        """
        super(RandomTemporalCrop, self).__init__(only_trial_data=True)
        assert 0 < max_crop_frac < 1
        self.max_crop_frac = max_crop_frac
        self.temporal_axis = temporal_axis

    def __call__(self, x, training=False):
        if not training:
            return x

        trial_len = x.shape[self.temporal_axis]
        crop_len = np.random.randint(int((1 - self.max_crop_frac) * trial_len), trial_len)
        offset = np.random.randint(0, trial_len - crop_len)

        return x[:, offset:offset + crop_len, ...]
