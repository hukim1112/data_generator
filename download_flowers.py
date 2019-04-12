import tensorflow as tf
import os
import sys
import dataset_util
import json
import urllib, tarfile
_DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'

def original_dataset_download_and_uncompress_tarball(dataset_dir, tarball_url=_DATA_URL):
    """Downloads the `tarball_url` and uncompresses it locally.

    Args:
      dataset_dir: The directory where the temporary files are stored.
      tarball_url: The URL of a tarball file.
    """
    origin_dir = os.path.join(dataset_dir, 'flower_photos')
    if not tf.gfile.Exists(origin_dir):
        tf.gfile.MakeDirs(origin_dir)

    if _dataset_exists(os.path.join(origin_dir)):
        print('Original dataset already exist. Downloading will not occur.')
        return
    filename = tarball_url.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dataset_dir)

def _dataset_exists(dataset_dir):
    sub_dir = os.listdir(dataset_dir)
    flower_category = set(
        ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])
    if flower_category.issubset(sub_dir):
        return True
    else:
        return False


if __name__ == "__main__":
    with open('configs/flowers.json') as file:
        config = json.load(file)

    dataset_dir = os.path.join('datasets', config['dataset_name'])
    original_dataset_download_and_uncompress_tarball(dataset_dir)
    origin_dir = os.path.join(dataset_dir, 'flower_photos')
    experiment_dir = os.path.join(dataset_dir, config['experiment_name'])
    dataset_util.make_splits_from_original_dataset(origin_dir, experiment_dir, config['splits'])