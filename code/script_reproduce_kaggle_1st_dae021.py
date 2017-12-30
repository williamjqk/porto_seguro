# %% this idea is from https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
base_path = '/home/tom/data/kaggle/porto_seguro_input'
data_path = '/home/tom/data/kaggle/porto_seguro_dae'

model_name = 'porto_seguro_dae021'

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


# %% build graph
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
in_dim = 221 # dimension of tfrecord features
BATCH_SIZE = dae_params['minibatch_size']
N_EPOCHS = dae_params['n_epochs']
learning_rate_decay = dae_params['learning_rate_decay']

import tensorflow as tf
tf.reset_default_graph()
def _parse_function(record):
    keys_to_features = {
        "rankgauss_feature": tf.FixedLenFeature((221,), tf.float32, default_value=tf.zeros((221,), dtype=tf.float32)),
        # "label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)
    return parsed_features["rankgauss_feature"]

filenames = [
    os.path.join(data_path, 'train_rankgauss.tfrecord'),
    os.path.join(data_path, 'test_rankgauss.tfrecord')
]

dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(buffer_size=200000)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.repeat(N_EPOCHS)
iterator = dataset.make_one_shot_iterator()
next_feature = iterator.get_next()



def gaussian_noise_layer(input_layer, std, name):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32, name=name)
    return input_layer + noise
def standard_layer(input_layer, n_nodes, std, keep_rate, bn_phase, layer_name, vscope_name=None):
    # with tf.variable_scope(vscope_name): # you can use scope on outside wrapper
    layer_o = tf.layers.dense(input_layer, n_nodes, name='{}_dense'.format(layer_name))
    # layer_o = tf.layers.batch_normalization(layer_o, name='{}_bn'.format(layer_name), training=bn_phase)
    layer_o = tf.nn.relu(layer_o, name='{}_relu'.format(layer_name))
    # layer_o = tf.nn.dropout(layer_o, keep_rate, name='{}_dropout'.format(layer_name))
    # layer_o = gaussian_noise_layer(layer_o, std, name='{}_gn'.format(layer_name))
    return layer_o


keep_rate = tf.placeholder_with_default(1.0, shape=())
input_swap_noise = tf.placeholder_with_default(0.0, shape=())
noise_std = tf.placeholder_with_default(0.0, shape=())
bn_phase = tf.placeholder_with_default(False, shape=()) # True for train, False for test(emmm...#TODO)

tf_x = next_feature
# tf_x = tf.placeholder(tf.float32, [None, in_dim]) #  X_train.shape
# tf_y = tf.placeholder(tf.int32, None)

with tf.variable_scope('denoise_autoencoder'):
    layer_noise = gaussian_noise_layer(tf_x, input_swap_noise, name='input_gn')
    layer1 = standard_layer(layer_noise, dae_params["layers"][0], noise_std, keep_rate, bn_phase, 'layer1')
    layer2 = standard_layer(layer1, dae_params["layers"][1], noise_std, keep_rate, bn_phase, 'layer2')
    layer3 = standard_layer(layer2, dae_params["layers"][2], noise_std, keep_rate, bn_phase, 'layer3')
    # semi_out = standard_layer(layer3, in_dim, noise_std, keep_rate, bn_phase, 'semi_out') # FIXME: just use dense is OK, shouldnot use dropout and relu
    semi_out = tf.layers.dense(layer3, in_dim, name='semi_out')

loss_semi = tf.losses.mean_squared_error(labels=tf_x, predictions=semi_out)

# optimizer = tf.train.RMSPropOptimizer(learning_rate = dae_params['learning_rate'])

global_step = tf.Variable(0, trainable=False)
initial_learning_rate = dae_params['learning_rate'] #初始学习率
learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=11625, # 11625 steps * 128 per batch
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
while True:
    try:
        result, loss_step, _, lr = sess.run([merge_op, loss_semi, train_op_semi, learning_rate],
                                {keep_rate: dae_params['keep_rate'],
                                 input_swap_noise: dae_params['input_swap_noise'],
                                 noise_std: dae_params['noise_std'],
                                 bn_phase: True})
        all_stage += 1
        if all_stage % 100 == 0:
            writer.add_summary(result, all_stage)
            print(f'lr {lr:.8f}, EPOCH {all_stage//11625}, all_stage {all_stage}, MSE {loss_step:.8f}')
    except tf.errors.OutOfRangeError:
        break


# %% save tf model, i should save model rather than hidden data
saver = tf.train.Saver()
saver.save(sess, os.path.join(model_path, 'dae_model'))
