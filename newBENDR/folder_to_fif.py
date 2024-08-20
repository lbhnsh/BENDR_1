def list_files_with_extension(root_folder, extension):
    assert isinstance(root_folder, str), "root_folder must be a string"
    assert isinstance(extension, str), "extension must be a string"
    assert os.path.isdir(root_folder), f"{root_folder} is not a valid directory"
    assert extension.startswith("."), "extension should start with a dot (.)"

    matching_files = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(extension):
                full_path = os.path.join(dirpath, filename)
                matching_files.append(full_path)

    assert isinstance(matching_files, list), "The output should be a list"

    return matching_files
