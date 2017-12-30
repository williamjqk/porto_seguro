# %% this idea is from https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
base_path = '/home/tom/data/kaggle/porto_seguro_input'

model_name = 'porto_seguro_dae00x5'
data_path = '/home/tom/data/kaggle/porto_seguro_dae'
model_path = os.path.join(data_path, model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)
train_filename = 'train_rankgauss.tfrecord'
test_filename = 'test_rankgauss.tfrecord'

import numpy as np
np.random.seed(20)
import pandas as pd

from tensorflow import set_random_seed

from sklearn.model_selection import StratifiedKFold

'''Data loading & preprocessing
'''

X_train_raw = pd.read_csv(os.path.join(base_path, 'train.csv'))
X_test_raw = pd.read_csv(os.path.join(base_path, 'test.csv'))

X_train_raw.columns

X_train, y_train = X_train_raw.iloc[:,2:], X_train_raw.target
X_test, test_id = X_test_raw.iloc[:,1:], X_test_raw.id

cols_use = [c for c in X_train.columns if not c.startswith('ps_calc_')]

print(f'len of cols_use: {len(cols_use)}')

X_train = X_train[cols_use]
X_test = X_test[cols_use]

cols_cat = {c: list(X_train[c].unique()) for c in X_train.columns if c.endswith('_cat')}
print(f'len of cols_cat: {len(cols_cat)}')

def one_hot_encoder(X_train, X_test, cols_cat):
    for c in cols_cat:
        for val in cols_cat[c]:
            newcol = c + '_oh_' + str(val)
            X_train[newcol] = (X_train[c].values == val).astype(np.int)
            X_test[newcol] = (X_test[c].values == val).astype(np.int)
        # NOTE: in mjahrer's approach, keep raw cat features in new X.
        # X_train.drop(labels=[c], axis=1, inplace=True)
        # X_test.drop(labels=[c], axis=1, inplace=True)
    return X_train,X_test

X_train, X_test = one_hot_encoder(X_train, X_test, cols_cat)
print(f'shapes of X_train, X_test: {X_train.shape, X_test.shape}')

X_train = X_train.fillna(-1)
X_test = X_test.fillna(-1)

X_0 = X_train.values
y_0 = y_train.values
X_1 = X_test.values

sub = X_test_raw['id'].to_frame()
sub['target'] = 0
sub_train = X_train_raw['id'].to_frame()
sub_train['target'] = 0


# %% 使用RankGauss处理非binary数据并存成h5
import os
import numpy as np
import pandas as pd
from scipy.special import erfinv
import matplotlib.pyplot as plt

def rank_gauss_old(x):
    # x is numpy vector
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2 # rank_x.max(), rank_x.min() should be in (-1, 1)
    efi_x = erfinv(rank_x) # np.sqrt(2)*erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x

def rank_gauss(x):
    dense_rank = rankdata(x, method='dense') - 1
    dense_rank_unique = np.unique(dense_rank)
    rg_unique = rank_gauss_old(dense_rank_unique)
    x2 = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        x2[i] = rg_unique[dense_rank[i]]

# os.path.join(data_path, train_filename)
# os.path.join(data_path, test_filename)

need_rankgauss_cols = []
for i in range(221):
    unique_values = np.unique(X_0[:,i])
    length = len(unique_values)
    if length > 2:
        print(f'{i}: {length}, max: {unique_values.max()}, min: {unique_values.min()}')
        need_rankgauss_cols.append(i)

X_all = np.vstack((X_0, X_1))
print(f'shape of X_all: {X_all.shape}')
for i in need_rankgauss_cols:
    x1 = X_all[:,i]
    X_all[:,i] = rank_gauss(x1)
    # pd.Series(rank_gauss(x1)).hist()
    # plt.show()

X_0 = X_all[:X_0.shape[0],:]
X_1 = X_all[X_0.shape[0]:,:]

# %% generate tfrecord data format
import tensorflow as tf
# generate trainset with label & features
tfrecord_filename = os.path.join(data_path, train_filename)
# tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_filename)
with tf.python_io.TFRecordWriter(tfrecord_filename) as tfrecord_writer:
    for i in range(len(y_0)):
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list = tf.train.Int64List(value=[y_0[i]])),
            'rankgauss_feature': tf.train.Feature(float_list = tf.train.FloatList(value=X_0[i].tolist()))
        }))
        tfrecord_writer.write(example.SerializeToString())

# generate tfrecord data of features for both train&test without label
tfrecord_filename = os.path.join(data_path, test_filename)
with tf.python_io.TFRecordWriter(tfrecord_filename) as tfrecord_writer:
    for i in range(X_1.shape[0]):
        example = tf.train.Example(features=tf.train.Features(feature={
            'rankgauss_feature': tf.train.Feature(float_list = tf.train.FloatList(value=X_1[i].tolist()))
        }))
        tfrecord_writer.write(example.SerializeToString())
