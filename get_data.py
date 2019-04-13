import os, sys
import json
import random
import tensorflow as tf
from preprocessing import inception_preprocessing
import dataset_util

root_dir = os.environ['DATA_GENERATOR']

def _parse_function(filename, label=None, is_training=True):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # image_resized = tf.image.resize_images(image_decoded, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    images = inception_preprocessing.preprocess_image(
        image_decoded, height=224, width=224, is_training=is_training)

    return images, label

def input_fn(filepaths, class_names_to_ids, batch_size, num_images, mode="training"):
    """An input function for training """

    # Convert the inputs to a Dataset
    dataset_filepath = tf.data.Dataset.from_tensor_slices(
        tf.cast(filepaths, tf.string))
    dataset_class = tf.data.Dataset.from_tensor_slices(
        [class_names_to_ids[os.path.basename(os.path.dirname(filepath))] for filepath in filepaths])
    dataset = tf.data.Dataset.zip((dataset_filepath, dataset_class))
    dataset = dataset.shuffle(num_images)
    if mode != "training":
        dataset = dataset.repeat(1)
    else:
        dataset = dataset.repeat()
    dataset = dataset.map(_parse_function, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2 * batch_size)
    return dataset

def inputpipeline_from_categorical_directories(config, batch_size, split, mode):
    dataset_name = config['dataset_name']
    experiment_name = config['experiment_name']
    experiment_dir = os.path.join(root_dir, 'datasets', dataset_name, experiment_name, split)
    filepaths, class_names_to_ids = dataset_util.get_filenames_and_class_to_id_dict(experiment_dir)
    dataset = input_fn(filepaths, class_names_to_ids, batch_size, len(filepaths), mode)
    iterator = dataset.make_one_shot_iterator()
    x, y = iterator.get_next()
    return x, y


def episodic_inputpipeline(config_name, n_way, n_shot, n_query):
    with open(os.path.join(root_dir, "configs", config_name+'.json'))
    train_split_path = os.path.join(root_dir, )


def get_data(config_name, batch_size, split, mode):
    with open(os.path.join(root_dir, "configs", config_name+'.json')) as file:
        config = json.load(file)

    if config['dataset_name'] == 'flowers':
        x, y = inputpipeline_from_categorical_directories(config, batch_size, split, mode)
        return x, y
    elif config['dataset_name'] == 'mnist':
        print("Not implemented yet")
    else:
        print("That config_name is not matched with any configuration.")