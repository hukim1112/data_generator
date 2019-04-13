# This code is referenced from github repo : https://github.com/cjalmeida/tensorflow-mnist/blob/master/tf_mnist/data.py
import tensorflow as tf
import numpy as np
import struct
import os
from pathlib import Path
from typing import Tuple


def normalize(images):
    """
    Normalize images to [-1,1]
    """

    images = tf.cast(images, tf.float32)
    images /= 255.
    images -= 0.5
    images *= 2
    return images


def transform_train(images, labels):
    """
    Apply transformations to MNIST data for use in training.
    To images: random zoom and crop to 28x28, then normalize to [-1, 1]
    To labels: one-hot encode.
    """
    zoom = 0.9 + np.random.random() * 0.2  # random between 0.9-1.1
    size = int(round(zoom * 28))
    print(images.shape)
    images = tf.image.resize_bilinear(images, (size, size))
    images = tf.image.resize_image_with_crop_or_pad(images, 28, 28)
    images = normalize(images)
    labels = tf.one_hot(labels, 10)
    labels = tf.squeeze(labels, 1)
    return images, labels


def transform_val(images, labels):
    """
    Normalize MNIST images and one-hot encode labels.
    """
    images = normalize(images)
    labels = tf.one_hot(labels, 10)
    labels = tf.squeeze(labels, 1)
    return images, labels


def read_mnist_images(IMAGES, split) -> np.ndarray:
    """
    Create tf.data.Dataset out of MNIST images data
    :param split: one of 'train' or 'val' for training or validation data
    """
    assert split in ['train', 'val']

    # read data as numpy array. The data structure is specified in Yann LeCun
    # site.
    fd = IMAGES[split].open('rb')
    magic, size, h, w = struct.unpack('>iiii', fd.read(4 * 4))
    data = np.frombuffer(fd.read(), 'u1').reshape(size, h, w, 1)
    fd.close()

    return data


def read_mnist_labels(LABELS, split) -> np.ndarray:
    """
    Create tf.data.Dataset out of MNIST labels data
    :param split: one of 'train' or 'val' for training or validation data
    """
    assert split in ['train', 'val']

    # read data as numpy array. The data structure is specified in Yann LeCun
    # site.
    fd = LABELS[split].open('rb')
    magic, size, = struct.unpack('>ii', fd.read(2 * 4))
    data = np.frombuffer(fd.read(), 'u1').reshape(size, 1)
    fd.close()

    return data


def dataset(config, batch_size, split) -> Tuple[tf.data.Dataset, int]:
    """
    Creates an Dataset for MNIST Data.
    This function create the correct tf.data.Dataset for a given split, transforms and
    batch inputs.
    """
    root_dir, dataset_name = config["root_dir"], config['dataset_name']
    dataset_dir = os.path.join(root_dir, 'datasets', dataset_name)

    INPUT = Path(dataset_dir)
    IMAGES = {'train': INPUT / 'train-images-idx3-ubyte',
              'val': INPUT / 't10k-images-idx3-ubyte'}
    LABELS = {'train': INPUT / 'train-labels-idx1-ubyte',
              'val': INPUT / 't10k-labels-idx1-ubyte'}

    images = read_mnist_images(IMAGES, split)
    labels = read_mnist_labels(LABELS, split)
    #random = np.random.RandomState(SEED)

    def gen():
        for image, label in zip(images, labels):
            yield image, label

    ds = tf.data.Dataset.from_generator(
        gen, (tf.uint8, tf.uint8), ((28, 28, 1), (1,)))

    if split == 'train':
        ds = ds.shuffle(512, seed=np.random.randint(0, 1024)).repeat()
        ds = ds.batch(batch_size).map(transform_train, num_parallel_calls=4)
        ds = ds.prefetch(2)
        return ds, len(labels)
    elif split == 'val':
        ds = ds.batch(batch_size).map(transform_val, num_parallel_calls=4)
        ds = ds.prefetch(2)
        return ds, len(labels)
