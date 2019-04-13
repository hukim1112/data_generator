import os
import tensorflow as tf
from preprocessing import inception_preprocessing
import dataset_util


def _parse_function(filename, label=None, is_training=True):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    images = inception_preprocessing.preprocess_image(
        image_decoded, height=224, width=224, is_training=is_training)
    return images, label


def input_fn(filepaths, class_names_to_ids, batch_size, num_images, is_training):
    """An input function for training """

    # Convert the inputs to a Dataset
    dataset_filepath = tf.data.Dataset.from_tensor_slices(
        tf.cast(filepaths, tf.string))
    dataset_class = tf.data.Dataset.from_tensor_slices(
        [class_names_to_ids[os.path.basename(os.path.dirname(filepath))] for filepath in filepaths])
    dataset = tf.data.Dataset.zip((dataset_filepath, dataset_class))
    dataset = dataset.shuffle(num_images)
    if is_training:
        dataset = dataset.repeat()
    else:
        dataset = dataset.repeat(1)

    dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=lambda x, y: _parse_function(
        x, y, is_training), num_parallel_calls=4, batch_size=batch_size))
    # dataset = dataset.map(lambda x, y: _parse_function(
    #     x, y, is_training), num_parallel_calls=4)
    # dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2 * batch_size)
    return dataset


def dataset(config, batch_size, split):
    if split == 'train':
        is_training = True
    root_dir, dataset_name, experiment_name = config[
        "root_dir"], config['dataset_name'], config['experiment_name']
    experiment_dir = os.path.join(
        root_dir, 'datasets', dataset_name, experiment_name, split)
    filepaths, class_names_to_ids = dataset_util.get_filenames_and_class_to_id_dict(
        experiment_dir)
    dataset = input_fn(filepaths, class_names_to_ids,
                       batch_size, len(filepaths), is_training)
    return dataset
