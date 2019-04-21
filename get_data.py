import os
import sys
import json
import random
import tensorflow as tf
from preprocessing import inception_preprocessing
import dataset_util
from input_fn import flowers, mnist
root_dir = os.environ['DATA_GENERATOR']

'''
This source codes define the way of input pipeline with high abstraction.
Now There are two options of categorical pipeline and episodic pipeline(will be implemented soon)
The first one is the categorical data pipeline get images and labels. It's usually used for classification and many unsupervised learnig models while labels are ignored. 
The second one is the episodic data pipeline(will be implemented soon). It choose N classes among all classes and sample K images from each category of N. We say this is N-way K-shot learning. It's used when our model is needed to solve classification of objects it has never seen during training phase. K-shot learning is known to be good for this problem.

In the near future, I will implement data pipeline to train our model with comparing image pairs.
This is good to build models for deep feature matching problems
'''


def categorical_inputpipeline(dataset):

    iterator = dataset.make_one_shot_iterator()
    x, y = iterator.get_next()
    return x, y


# def episodic_inputpipeline(config_name, n_way, n_shot, n_query):
#     with open(os.path.join(root_dir, "configs", config_name + '.json'))
#     train_split_path = os.path.join(root_dir, )


def data_pipeline(config_name, batch_size, split, mode=None):
    ''' 
    Get inputpipeline with a dataset name. 
    If you want to know the ways to build each dataset by tf.data.dataset API, Check the 'input_fn' folder.
    There are few parameters to controll each dataset pipeline and They are defined on the configuration file of each dataset(at ./configs).
    '''
    with open(os.path.join(root_dir, "configs", config_name + '.json')) as file:
        config = json.load(file)
    config["root_dir"] = root_dir
    if config['dataset_name'] == 'flowers':
        dataset = flowers.dataset(config, batch_size, split)
        x, y = categorical_inputpipeline(dataset)
        return x, y
    elif config['dataset_name'] == 'mnist':
        dataset, num_data = mnist.dataset(config, batch_size, split)
        x, y = categorical_inputpipeline(dataset)
        return x, y
    else:
        print("That config_name is not matched with any configuration.")
