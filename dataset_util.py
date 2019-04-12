import os, sys
import random
import shutil

def _get_filenames_and_classes(dataset_dir):
    directories = []
    class_names = []
    for dir_name in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, dir_name)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(dir_name)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)

    return photo_filenames, sorted(class_names)

def get_filenames_and_class_to_id_dict(dataset_dir):
	# input : path to dataset dir
	# output : return the list of file paths and dictionary of category name(string) to class id(integer)
    filepaths, class_names = _get_filenames_and_classes(dataset_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    return filepaths, class_names_to_ids

def split_list_with_index(_list, amount):
    return _list[:amount], _list[amount:]

def make_splits_from_original_dataset(origin, dest, dict_of_split_info):
    # Split original dataset into splits(e.g train, eval, test)
    """
        arguments
          1. origin : path to original datset(not splited)
          2. dest : destination path(the root director includes train, eval and test directories)
          2. dict_of_split_info : dict of split directory names and number of images each split each split.
              example : {'train' : 2500, 'eval' : 500, 'test' : 670}. note that all files are 3670.
        return
          None but images in original directory would be splited into directories of split names     
    """
    # Load dataset from original flower-image names and shuffle them.

    if os.path.exists(dest):
    	print("That dataset for experiment name already exists, Please check out config file again")
    	return

    filepaths, class_names_to_ids = get_filenames_and_class_to_id_dict(origin)
    random.shuffle(filepaths)

    splited_list = {}
    for split_name in dict_of_split_info.keys():
        splited_list[split_name], filepaths = split_list_with_index(
            filepaths, dict_of_split_info[split_name])

    # Check total numbers in dictionary to be matched with real total number of files.
    if len(filepaths) != 0:
        print("total number of images is not correct in your split_num dictionary.")
        return

    # Make directories for each category of each split
    for split_name in dict_of_split_info.keys():
        for category in class_names_to_ids.keys():
            os.makedirs(os.path.join(
                dest, split_name, category), exist_ok=True)

    # Copy src_file in original directory to dest separated by splits("train", "eval", "test")
    for split_name in splited_list.keys():
        path = os.path.join(dest, split_name)
        for src_file in splited_list[split_name]:
            filename = os.path.basename(src_file)
            category = os.path.split(os.path.dirname(src_file))[1]
            dest_file = os.path.join(path, category, filename)
            shutil.copyfile(src_file, dest_file)
    return