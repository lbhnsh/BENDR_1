import torch
import argparse
from src.bendr import BendingCollegeWav2Vec, ConvEncoderBENDR, BENDRContextualizer, BENDRClassifier
from src.multiview import finetune, evaluate_classifier
from src.eegdataset import construct_eeg_datasets
from torch.optim import AdamW
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os
import datetime
import wandb
from src.batch import RandomTemporalCrop


def check_output_path(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        i = 1
        while os.path.exists(output_path + f'_v_{i}'):
            i += 1
        output_path = output_path + f'_v_{i}'
        os.makedirs(output_path, exist_ok=True)
    return output_path


mul_channel_explanations = {
    'None': 'Multi channel setup is set to None.',
    'sample_channel': 'Multi channel setup is set to sample_channel. This means that sampled channels will be used as each others augmented versions.',
    'avg_ch': 'Multi channel setup is set to ch_avg. This means that the channels are averaged before convolutions.'
}

train_mode = 'pretrain'
standardize_epochs = 'channelwise'
bendr_setup = True
hidden_size = 256
out_dim = 64
mask_rate = 0.065
mask_span = 10
layer_drop = 0.01
multi_gpu = True
temp = 0.1
encoder_grad_frac = 0.1
num_negatives = 20
enc_feat_l2 = 1.0
lr = 0.00002
weight_decay = 0.01
betas = [0.9, 0.98]

epochs = 50
num_workers = 6
resume = None
validation_interval = 100
train_log_interval = 100
batch_size = 64
warmup_frac = 0.05

mask_threshold = 0.85
mask_inflation = 1.
mask_pct_max = 0.6
chunk_duration = 30
upsample_crop = 32
batch_crop_frac = 0.05

output_path = "/home/zeydabadi/my_projects/newBENDR/output/"
print('Saving outputs in', output_path)


encoder = ConvEncoderBENDR(6, encoder_h=hidden_size, out_dim=out_dim)
contextualizer = BENDRContextualizer(out_dim, layer_drop=0.01)
# add arguments from BENDR config
experiment = {
    'mask_threshold': 0.85,
    'mask_inflation': 1.,
    'mask_pct_max': 0.65
}
bending_college_args = {
    "mask_rate": mask_rate,
    "mask_span": mask_span,
    "layer_drop": 0.01,
    "multi_gpu": True,
    "temp": 0.1,
    "encoder_grad_frac": 0.1,
    "num_negatives": num_negatives,
    "enc_feat_l2": 1.0
}
optimizer_params = {
    "lr": 0.00002,
    "weight_decay": 0.01,
    "betas": [0.9, 0.98]
}
augmentation_params = {
    "upsample_crop": 32,
    "batch_crop_frac": 0.05
}
training_params = {
    "epochs": epochs,
    "validation_interval": 100,
    "train_log_interval": 100,
    "batch_size": 64,
    "warmup_frac": 0.05
}
process = BendingCollegeWav2Vec(
    encoder, contextualizer, **bending_college_args)

# Slower learning rate for the encoder
process.set_optimizer(torch.optim.Adam(
    process.parameters(), **optimizer_params))


process.add_batch_transform(RandomTemporalCrop(
    max_crop_frac=augmentation_params["batch_crop_frac"]))


def epoch_checkpoint(metrics):
    encoder.save('checkpoints/encoder_epoch_{}.pt'.format(metrics['epoch']))
    contextualizer.save(
        'checkpoints/contextualizer_epoch_{}.pt'.format(metrics['epoch']))


def simple_checkpoint(metrics):
    if metrics is not None and metrics['Accuracy'] > experiment['mask_threshold'] and \
            metrics['Mask_pct'] < experiment['mask_pct_max']:
        process.mask_span = int(
            process.mask_span * experiment['mask_inflation'])

    encoder.save('checkpoints/encoder.pt')
    contextualizer.save('checkpoints/contextualizer.pt')


process.fit(pretrain_loader, epoch_callback=epoch_checkpoint, num_workers=num_workers,
            validation_dataset=pretrain_val_loader, resume_epoch=resume, log_callback=simple_checkpoint,
            **training_params)

print(process.evaluate(pretrain_val_loader))

encoder.save('checkpoints/encoder_best_val.pt')
contextualizer.save('checkpoints/contextualizer_best_val.pt')
