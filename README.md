
#####  1. INSTALL DEPENDENCIES
pip install -r requirements.txt

##### 2. add the path of data_generator folder(root directory of repository) into .bashrc
e.g : export DATA_GENERATOR=/home/kerry/prj/data_generator

And $source .bashrc

##### 3. Download a dataset you want to use.

e.g python download_mnist.py

##### 4. Usage

```
import os, sys

root_dir = os.environ['DATA_GENERATOR']

sys.path.append(root_dir)

import get_data

images, labels = get_data.data_pipeline("mnist", 128, 'train') #(dataset name, batch size, split)


```

##### 5. Reference of tf.data
1) https://www.tensorflow.org/api_docs/python/tf/data
2) https://www.tensorflow.org/guide/datasets
3) https://medium.com/ymedialabs-innovation/how-to-use-dataset-and-iterators-in-tensorflow-with-code-samples-3bb98b6b74ab
