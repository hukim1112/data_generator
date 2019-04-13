from pathlib import Path
from hashlib import md5
import gzip
import shutil
import numpy as np
import tensorflow as tf
import os, json
import requests
from urllib.parse import urlparse

from tqdm import tqdm

    
FILES_GZ = [
    ('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
     '6bbc9ace898e44ae57da46a324031adb'),
    ('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
     'a25bea736e30d166cdddb491f175f624'),
    ('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
     '2646ac647ad5339dbf082846283269ea'),
    ('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
     '27ae3e4e09519cfbb04c329615203637')
]

def md5sum(file: Path):
    data = file.open('rb').read()
    return md5(data).hexdigest()

def dataset_download(dataset_dir, **kwargs):
    """
    Get MNIST data from Yann LeCun site. Check for existence first.
    """
    INPUT = Path(dataset_dir)

    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    for raw_url, file_hash in FILES_GZ:
        url = urlparse(raw_url)
        # store data in INPUT
        dest = INPUT / Path(url.path).name

        # check if we already have the unpacked data
        dest_unpacked = dest.with_suffix('')
        if dest_unpacked.exists() and md5sum(dest_unpacked) == file_hash:
            tqdm.write(f'Already downloaded {dest_unpacked}')
            continue

        # do download with neat progress bars
        r = requests.get(raw_url, stream=True)
        file_size = int(r.headers.get('content-length', 0))
        tqdm.write(f'Downloading {raw_url}')
        if file_size:
            bar = tqdm(total=file_size)
        else:
            bar = tqdm()
        with dest.open('wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    bar.update(len(chunk))
        bar.close()

        # use gzip module to unpack downloaded files
        tqdm.write(f'Unpacking {dest}')
        with gzip.open(str(dest), 'rb') as gz_src:
            with dest_unpacked.open('wb') as gz_dst:
                shutil.copyfileobj(gz_src, gz_dst)

        dest.unlink()


if __name__ == "__main__":
    with open('configs/mnist.json') as file:
        config = json.load(file)

    dataset_dir = os.path.join('datasets', config['dataset_name'])
    dataset_download(dataset_dir)