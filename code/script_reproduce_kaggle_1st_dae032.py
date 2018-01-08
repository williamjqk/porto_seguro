# %% this idea is from https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
base_path = '/home/tom/data/kaggle/porto_seguro_input'
data_path = '/home/tom/data/kaggle/porto_seguro_dae'
# base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
# data_path = '/home/ljc/data/porto_seguro_dae'

model_name = 'porto_seguro_dae032'

model_path = os.path.join(data_path, model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)

dae_params = {
    "layers": [1500, 1500, 1500],
    "learning_rate": 3e-3, # 3e-3,
    "minibatch_size": 128,
    "learning_rate_decay": 0.995,
    "keep_rate": 1.0,#0.8,
    "input_swap_noise": 0.15,
    "noise_std": 0.0,
    "n_epochs": 1000 # 1000
}


# %%

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

def rank_gauss(x):
    # x is numpy vector
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2 # rank_x.max(), rank_x.min() should be in (-1, 1)
    efi_x = erfinv(rank_x) # np.sqrt(2)*erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x

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

# %%
import copy
import time
tic = time.time()
for i in range(100):
    X_batch = X_all[np.random.randint(X_all.shape[0], size=128), :]
    # n_features = [i for i in range(X_0.shape[1])]
    # swap_cols = np.random.randint(X_0.shape[1], size=15)

    X_batch_noise = np.zeros(X_batch.shape)
    for i in range(X_batch.shape[0]):
        swap_cols = np.random.randint(X_all.shape[1], size=15)
        one_row = copy.deepcopy(X_batch[i,:])
        for i2 in swap_cols:
            one_row[i2] = X_all[np.random.choice(X_all.shape[0]),i2]
        X_batch_noise[i,:] = one_row
print(f'{time.time() - tic}s per 100 batches') # 5.8s per 1000 batches
# print(X_batch)
# print(X_batch_noise)

# %%
import threading
import multiprocessing as mp
# mutex = threading.Lock()
# mutex = mp.Lock()


from multiprocessing.pool import Pool
from multiprocessing import Process
from queue import Queue
# from multiprocessing.queues import Queue
batch_q = Queue(maxsize=10000)
n_swaps = int(X_all.shape[1] * dae_params["input_swap_noise"])
def func_put_q():
    while True:
        if not batch_q.full():
            X_batch = X_all[np.random.randint(X_all.shape[0], size=dae_params['minibatch_size']), :]
            # n_features = [i for i in range(X_0.shape[1])]
            # swap_cols = np.random.randint(X_0.shape[1], size=15)

            X_batch_noise = np.zeros(X_batch.shape)
            for i in range(X_batch.shape[0]):
                swap_cols = np.random.randint(X_all.shape[1], size=n_swaps)
                one_row = copy.deepcopy(X_batch[i,:])
                for i2 in swap_cols:
                    one_row[i2] = X_all[np.random.choice(X_all.shape[0]),i2]
                X_batch_noise[i,:] = one_row

            b_dict = {'x_b_noise': X_batch_noise, 'x_b_raw':X_batch}
            batch_q.put(b_dict)

# def batch_gen_func():
#     while True:
#         X_batch = X_all[np.random.randint(X_all.shape[0], size=128), :]
#         # n_features = [i for i in range(X_0.shape[1])]
#         # swap_cols = np.random.randint(X_0.shape[1], size=15)
#
#         X_batch_noise = np.zeros(X_batch.shape)
#         for i in range(X_batch.shape[0]):
#             swap_cols = np.random.randint(X_all.shape[1], size=n_swaps)
#             one_row = copy.deepcopy(X_batch[i,:])
#             for i2 in swap_cols:
#                 one_row[i2] = X_all[np.random.choice(X_all.shape[0]),i2]
#             X_batch_noise[i,:] = one_row
#
#         b_dict = {'x_b_noise': X_batch_noise, 'x_b_raw':X_batch}
#         yield b_dict

thread1 = threading.Thread(target=func_put_q)
thread1.daemon = True
thread1.start()

# import QPhantom
# from QPhantom.core.data import DataQueue
# batch_queue = DataQueue(batch_gen_func, capacity=1024, num_worker=8)
# batch_queue.start()

# p = Process(target=func_put_q)
# p.daemon = True
# p.start()

# for i in range(5):
#     thread1 = threading.Thread(target=func_put_q)
#     thread1.daemon = True
#     thread1.start()

# for i in range(3):
#     p = Process(target=func_put_q, args=(mutex,))
#     p.daemon = True
#     p.start()

# batch_q.qsize()
# %% The last cell preprocess data. In this cell, let data go into one model
import os
import pandas as pd
import numpy as np


data_root = model_path
model_root = os.path.join(data_root, "test_model")#"/tmp/test_model"


# %% build graph

in_dim = 221 # dimension of tfrecord features
BATCH_SIZE = dae_params['minibatch_size']
N_EPOCHS = dae_params['n_epochs']
learning_rate_decay = dae_params['learning_rate_decay']
steps_per_epoch = X_all.shape[0] // BATCH_SIZE

import tensorflow as tf
tf.reset_default_graph()
# def _parse_function(record):
#     keys_to_features = {
#         "rankgauss_feature": tf.FixedLenFeature((221,), tf.float32, default_value=tf.zeros((221,), dtype=tf.float32)),
#         # "label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
#     }
#     parsed_features = tf.parse_single_example(record, keys_to_features)
#     return parsed_features["rankgauss_feature"]
#
# filenames = [
#     # os.path.join(data_path, 'train_rankgauss.tfrecord'),
#     # os.path.join(data_path, 'test_rankgauss.tfrecord')
#     os.path.join(data_path, 'train_rankgauss_porto_seguro_dae00x2.tfrecord'),
#     os.path.join(data_path, 'test_rankgauss_porto_seguro_dae00x2.tfrecord')
# ]
#
# dataset = tf.data.TFRecordDataset(filenames)
# dataset = dataset.map(_parse_function)
# dataset = dataset.shuffle(buffer_size=200000)
# dataset = dataset.batch(BATCH_SIZE)
# dataset = dataset.repeat(N_EPOCHS)
# iterator = dataset.make_one_shot_iterator()
# next_feature = iterator.get_next()



def gaussian_noise_layer(input_layer, std, name):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32, name=name)
    return input_layer + noise
def standard_layer(input_layer, n_nodes, std, keep_rate, bn_phase, layer_name, vscope_name=None):
    # with tf.variable_scope(vscope_name): # you can use scope on outside wrapper
    layer_o = tf.layers.dense(input_layer, n_nodes, name='{}_dense'.format(layer_name))
    layer_o = tf.layers.batch_normalization(layer_o, name='{}_bn'.format(layer_name), training=bn_phase)
    layer_o = tf.nn.relu(layer_o, name='{}_relu'.format(layer_name))
    layer_o = tf.nn.dropout(layer_o, keep_rate, name='{}_dropout'.format(layer_name))
    layer_o = gaussian_noise_layer(layer_o, std, name='{}_gn'.format(layer_name))
    return layer_o


keep_rate = tf.placeholder_with_default(1.0, shape=())
input_swap_noise = tf.placeholder_with_default(0.0, shape=())
noise_std = tf.placeholder_with_default(0.0, shape=())
bn_phase = tf.placeholder_with_default(False, shape=()) # True for train, False for test(emmm...#TODO)
x_b_noise = tf.placeholder(tf.float32, shape=[None,in_dim], name='x_b_noise')
x_b_raw = tf.placeholder(tf.float32, shape=[None,in_dim],  name='x_b_raw')
# tf_x = next_feature
# tf_x = tf.placeholder(tf.float32, [None, in_dim]) #  X_train.shape
# tf_y = tf.placeholder(tf.int32, None)

with tf.variable_scope('denoise_autoencoder'):
    layer_noise = x_b_noise
    # layer_noise = gaussian_noise_layer(tf_x, input_swap_noise, name='input_gn')
    layer1 = standard_layer(layer_noise, dae_params["layers"][0], noise_std, keep_rate, bn_phase, 'layer1')
    layer2 = standard_layer(layer1, dae_params["layers"][1], noise_std, keep_rate, bn_phase, 'layer2')
    layer3 = standard_layer(layer2, dae_params["layers"][2], noise_std, keep_rate, bn_phase, 'layer3')
    # semi_out = standard_layer(layer3, in_dim, noise_std, keep_rate, bn_phase, 'semi_out') # FIXME: just use dense is OK, shouldnot use dropout and relu
    semi_out = tf.layers.dense(layer3, in_dim, name='semi_out')

loss_semi = tf.losses.mean_squared_error(labels=x_b_raw, predictions=semi_out)

# optimizer = tf.train.RMSPropOptimizer(learning_rate = dae_params['learning_rate'])

global_step = tf.Variable(0, trainable=False)
initial_learning_rate = dae_params['learning_rate'] #初始学习率
learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=steps_per_epoch, # 11625 steps * 128 per batch
                                           decay_rate=learning_rate_decay)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

add_global = global_step.assign_add(1)
# with tf.control_denpendices([add_global]):
#     train_op = opt.minimise(loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops+[add_global]):
# with tf.control_dependencies(update_ops):
    train_op_semi = optimizer.minimize(loss_semi)
# train_op_semi = optimizer.minimize(loss_semi)

tf.summary.scalar('loss', loss_semi)
tf.summary.histogram('layer1', layer1)
tf.summary.histogram('layer2', layer2)
tf.summary.histogram('layer3', layer3)
merge_op = tf.summary.merge_all()

# %% begin to train
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# sess = tf.train.MonitoredTrainingSession()
sess = tf.Session()
sess.run(init_op)     # initialize var in graph
writer = tf.summary.FileWriter( os.path.join(model_path, 'log'), sess.graph)     # write to file

all_stage = 0
# while True:
for i in range(steps_per_epoch * N_EPOCHS):
    try:
        next_batch = batch_q.get()
        # next_batch = next(batch_queue.buffer())
        result, loss_step, _, lr = sess.run([merge_op, loss_semi, train_op_semi, learning_rate],
                                {x_b_noise: next_batch['x_b_noise'],
                                 x_b_raw: next_batch['x_b_raw'],
                                 keep_rate: dae_params['keep_rate'],
                                 # input_swap_noise: dae_params['input_swap_noise'],
                                 noise_std: dae_params['noise_std'],
                                 bn_phase: True})
        all_stage += 1
        if all_stage % 100 == 0:
            writer.add_summary(result, all_stage)
            print(f'lr {lr:.8f}, EPOCH {all_stage//steps_per_epoch}, all_stage {all_stage}, MSE {loss_step:.8f}')
        if all_stage % 1000 == 0:
            print(f'QSZIE: {batch_q.qsize()}')
    except tf.errors.OutOfRangeError:
        break


# %% save tf model, i should save model rather than hidden data
saver = tf.train.Saver()
saver.save(sess, os.path.join(model_path, 'dae_model'))
