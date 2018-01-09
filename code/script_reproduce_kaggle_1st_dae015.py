# %% this idea is from https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
base_path = '/home/tom/mywork/some_test/porto_seguro/input/'

model_name = 'porto_seguro_dae015'
data_path = '/home/tom/data/porto_seguro_dae'
model_path = os.path.join(data_path, model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# %% The last cell preprocess data. In this cell, let data go into one model
# in dae003, try hyperopt
import os
import pandas as pd
import numpy as np
import hyperopt as hp
from hyperopt import fmin, rand, tpe, hp, pyll, base, space_eval, Trials
from hyperopt.pyll import Apply, as_apply, dfs, scope
from hyperopt.mongoexp import MongoTrials
from PPMoney.core.utils import proc_run
from PPMoney.core.model.space import space_lgb_binary, space_update, sample_int
from PPMoney.core.data import HDFDataSet
suggest = rand.suggest

data_root = model_path
model_root = os.path.join(data_root, "test_model")#"/tmp/test_model"

from PPMoney.core.data import HDFDataSet
file_tr = os.path.join(data_path, 'mjahrer_1st_train.dataset')
dataset_tr = HDFDataSet(file_tr, chunk_size=2048)
X_0 = dataset_tr['quantile_feature']
y_0 = dataset_tr['label']

file_t = os.path.join(data_path, 'mjahrer_1st_test.dataset')
dataset_t = HDFDataSet(file_t, chunk_size=2048)
X_1 = dataset_t['quantile_feature']

X_all = np.vstack((X_0, X_1))
print(X_all.shape)

# %% generate tfrecord data format
import tensorflow as tf
# generate trainset with label & features
tfrecord_filename = os.path.join(data_path, 'train_quantile_features.tfrecord')
# tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_filename)
with tf.python_io.TFRecordWriter(tfrecord_filename) as tfrecord_writer:
    for i in range(len(y_0)):
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list = tf.train.Int64List(value=[y_0[i]])),
            'quantile_feature': tf.train.Feature(float_list = tf.train.FloatList(value=X_0[i].tolist()))
        }))
        tfrecord_writer.write(example.SerializeToString())

# generate tfrecord data of features for both train&test without label
tfrecord_filename = os.path.join(data_path, 'test_quantile_features.tfrecord')
with tf.python_io.TFRecordWriter(tfrecord_filename) as tfrecord_writer:
    for i in range(X_1.shape[0]):
        example = tf.train.Example(features=tf.train.Features(feature={
            'quantile_feature': tf.train.Feature(float_list = tf.train.FloatList(value=X_1[i].tolist()))
        }))
        tfrecord_writer.write(example.SerializeToString())

# %% read some (feature,label) from tfrecord
def _parse_function(record):
    keys_to_features = {
        "quantile_feature": tf.FixedLenFeature((221,), tf.float32, default_value=tf.zeros((221,), dtype=tf.float32)),
        "label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)
    return parsed_features["quantile_feature"], parsed_features["label"]

filenames = [
    os.path.join(data_path, 'train_quantile_features.tfrecord'),
    # os.path.join(data_path, 'test_quantile_features.tfrecord')
]

dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(buffer_size=100000)
dataset = dataset.batch(128)
dataset = dataset.repeat(10)
iterator = dataset.make_one_shot_iterator()
next_feature, next_label = iterator.get_next()

sess = tf.Session()

sess.run([next_feature, next_label])

# %% read some (feature,) from tfrecord
tf.reset_default_graph()
def _parse_function(record):
    keys_to_features = {
        "quantile_feature": tf.FixedLenFeature((221,), tf.float32, default_value=tf.zeros((221,), dtype=tf.float32)),
        # "label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)
    return parsed_features["quantile_feature"]

filenames = [
    os.path.join(data_path, 'train_quantile_features.tfrecord'),
    os.path.join(data_path, 'test_quantile_features.tfrecord')
]

dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(buffer_size=100000)
dataset = dataset.batch(128)
dataset = dataset.repeat(10)
iterator = dataset.make_one_shot_iterator()
next_feature = iterator.get_next()

sess = tf.Session()

sess.run([next_feature, ])
