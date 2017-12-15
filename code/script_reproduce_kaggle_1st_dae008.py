# %% this idea is from https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'

model_name = 'porto_seguro_dae008'
data_path = '/home/ljc/data/porto_seguro_dae'
model_path = os.path.join(data_path, model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)

from collections import Counter
import numpy as np
np.random.seed(20)
import pandas as pd
from PPMoney.core.data import HDFDataSet
from tensorflow import set_random_seed

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
nfold = 5
skf = StratifiedKFold(n_splits=nfold, random_state=0)

'''Data loading & preprocessing
'''

dataset_tr = HDFDataSet(os.path.join(data_path, 'mjahrer_1st_train.dataset'), chunk_size=2048)
dataset_t = HDFDataSet(os.path.join(data_path, 'mjahrer_1st_test.dataset'), chunk_size=2048)
X_0 = dataset_tr['dae_hidden_feature']
y_0 = dataset_tr['label']
X_1 = dataset_t['dae_hidden_feature']
print(f'shapes of X_0, X_1: {X_0.shape, X_1.shape}')

X_train_raw = pd.read_csv(base_path+'train.csv')
X_test_raw = pd.read_csv(base_path+'test.csv')
sub = X_test_raw['id'].to_frame()
sub['target'] = 0
sub_train = X_train_raw['id'].to_frame()
sub_train['target'] = 0

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
import tensorflow as tf
tf.reset_default_graph()

dae_params = {
    "layers": [1000, 1000],
    "learning_rate": 1e-3, # 3e-3,
    "minibatch_size": 128,
    "learning_rate_decay": 0.995,
    "keep_rate": 0.7,
    "input_swap_noise": 0.15,
    "noise_std": 0.0,
    "n_epochs": 1000 # 1000
}

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

in_dim = X_0.shape[1]
BATCH_SIZE = dae_params['minibatch_size']

keep_rate = tf.placeholder_with_default(1.0, shape=())
input_swap_noise = tf.placeholder_with_default(0.0, shape=())
noise_std = tf.placeholder_with_default(0.0, shape=())
bn_phase = tf.placeholder_with_default(False, shape=()) # True for train, False for test(emmm...#TODO)
tf_x = tf.placeholder(tf.float32, [None, in_dim]) #  X_train.shape
tf_y = tf.placeholder(tf.int32, None)

with tf.variable_scope('supervised_nn'):
    layer1 = standard_layer(tf_x, dae_params["layers"][0], noise_std, keep_rate, bn_phase, 'layer1')
    layer2 = standard_layer(layer1, dae_params["layers"][1], noise_std, keep_rate, bn_phase, 'layer2')
    output = tf.layers.dense(layer2, 2)                     # output laye

# a1 = tf.placeholder(tf.int32, 5)
# a2 = tf.placeholder(tf.float32, [5, 2])
# a3 = tf.losses.sparse_softmax_cross_entropy(labels=a1, logits=a2)
# tf.__version__

y_pred_v = tf.nn.softmax(output)
loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=output)

# loss = tf.losses.sparse_softmax_cross_entropy(labels=[1,2], logits=np.array([[1.0,0.0],[0.2,0.8]]))

optimizer = tf.train.AdamOptimizer(learning_rate = dae_params['learning_rate'])

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)

sess = tf.Session()                                                                 # control training and others
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)     # initialize var in graph

tf.summary.scalar('loss', loss)
tf.summary.histogram('layer1', layer1)
tf.summary.histogram('layer2', layer2)

writer = tf.summary.FileWriter( os.path.join(model_path, 'log'), sess.graph)     # write to file
merge_op = tf.summary.merge_all()
# %% begin to train
def batch_generator(X, y, batch_size=64):
    N = X.shape[0]
    m = batch_size
    steps = (N-1)//m + 1
    pad_size = steps*m - N
    # Xpad = np.vstack(X, randomchoose(X, padsize))
    while True:
        for i in range(steps):
            X_batch = X[(m*i):(m*(i+1)), :]
            y_batch = y[(m*i):(m*(i+1))]
            if pad_size > 0 and i == steps-1:
                rand_idx = np.random.choice(N, pad_size)
                X_batch = np.vstack((X_batch, X[rand_idx]))
                y_batch = np.hstack((y_batch, y[rand_idx]))
            yield X_batch,y_batch

shuffle_random_range = np.arange(X_0.shape[0])
for i in range(6):
    np.random.shuffle(shuffle_random_range)
X_0 = X_0[shuffle_random_range]

def predict_by_minibatch(X_input, batch_size):
    y_out_stack = np.zeros((X_input.shape[0],))
    i = 0
    while True:
        X_input_batch = X_input[i*batch_size:(i+1)*batch_size,:]
        y_out_stack[i*batch_size:(i+1)*batch_size] += sess.run(y_pred_v, {tf_x: X_input_batch})[:,1]
        i += 1
        if i*batch_size >= X_input.shape[0]:
            break
    return y_out_stack

def balance_train_data(X_0, y_0):
    Counter(y_0)
    counter1 = Counter(y_0)
    counter1_l = sorted(counter1, key=counter1.get)
    balance_d = [counter1_l[0], counter1[counter1_l[-1]]//counter1[counter1_l[0]]]

    mask_0 = (y_0 == balance_d[0])
    X_0.shape
    X_0_ = np.tile(X_0[mask_0], (balance_d[1],1))
    X_0_.shape
    y_0.shape
    y_0_ = np.tile(y_0[mask_0], balance_d[1])
    y_0_.shape
    X_tf = np.vstack((X_0, X_0_))
    y_tf = np.hstack((y_0, y_0_))

    shuffle_random_range = np.arange(X_tf.shape[0])
    for i in range(6):
        np.random.shuffle(shuffle_random_range)

    X_tf = X_tf[shuffle_random_range]
    y_tf = y_tf[shuffle_random_range]

    print(f'X_tf.shape, y_tf.shape: {X_tf.shape, y_tf.shape}')
    return X_tf, y_tf

def ks_score(label, preds):
    fpr, tpr, thresholds = roc_curve(label, preds)
    ks = max(tpr-fpr)
    return ks

def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)

def gini_mlp(pred, y):
    return gini(y, pred) / gini(y, y)

import time
y_pred = np.zeros(X_1.shape[0])
saver = tf.train.Saver()

batch_size = dae_params['minibatch_size']
n_steps = X_0.shape[0] // batch_size
n_steps *= 1000#1

cv_gini_score = 0
cv_gini_score_l = []
fold_step_l = []
for i1, (train_index, test_index) in enumerate(skf.split(X_0,y_0)):
    t0 = time.time()
    # train_index, test_index = tmp_index[0]
    print(f"THE {i1}th fold, ", "TRAIN:", train_index.shape, "TEST:", test_index.shape)
    X_train_0, X_test = X_0[train_index], X_0[test_index]
    y_train_0, y_test = y_0[train_index], y_0[test_index]

    X_train,y_train = balance_train_data(X_train_0, y_train_0)
    # X_train,y_train = balance_train_data(X_train_0, y_train_0)
    # X_train,y_train = (X_train_0, y_train_0)

    batch_gen = batch_generator(X_train, y_train, batch_size=batch_size)
    print('define a batch generator')
    eval_ks_l = []
    gini_eval_l = []

    sess = tf.Session()                                                                 # control training and others
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)     # initialize var in graph

    for step in range(n_steps):
        # train and net output
        X_batch, y_batch = next(batch_gen)
        # _, acc, pred = sess.run([train_op, accuracy, output], {tf_x: X_batch, tf_y: y_batch})
        # _, pred = sess.run([train_op, output], {tf_x: X_batch, tf_y: y_batch})

        result, loss_step, _ = sess.run([merge_op, loss, train_op],
                                {tf_x: X_batch,
                                 tf_y: y_batch,
                                 keep_rate: dae_params['keep_rate'],
                                 input_swap_noise: dae_params['input_swap_noise'],
                                 noise_std: dae_params['noise_std'],
                                 bn_phase: True})

        if step % 100 == 0:
            writer.add_summary(result, step)

            y_train_pred = predict_by_minibatch(X_train_0, 20000)
            # y_train_pred = sess.run(y_pred_v, {tf_x: X_train_0})[:,1] # FIXME: X_train_0 is too large
            auc_train = roc_auc_score(y_train_0, y_train_pred)
            # ks_train = ks_score(y_train_0, y_train_pred)
            gini_train = gini_mlp(y_train_pred, y_train_0)
            # y_eval_pred = sess.run(y_pred_v, {tf_x: X_test})[:,1]
            y_eval_pred = predict_by_minibatch(X_test, 20000)
            auc_eval = roc_auc_score(y_test, y_eval_pred)
            # ks_eval = ks_score(y_test, y_eval_pred)
            gini_eval = gini_mlp(y_eval_pred, y_test)

            gini_eval_l.append(gini_eval)
            if np.argmax(gini_eval_l) == len(gini_eval_l)-1:
                saver.save(sess, os.path.join(model_path,'model.ckpt'))

            if len(gini_eval_l) - np.argmax(gini_eval_l) > 50:
                print(f"Best step {np.argmax(gini_eval_l)*100} best EVAL gini score: {max(gini_eval_l):.6f}")
                print(f"{step} TRAIN auc: {auc_train:.6f}, gini: {gini_train:.6f}; EVAL auc: {auc_eval:.6f}, gini: {gini_eval:.6f}")
                break

            if step % 100 == 0:
                print(f"{step} TRAIN auc: {auc_train:.6f}, gini: {gini_train:.6f}; EVAL auc: {auc_eval:.6f}, gini: {gini_eval:.6f}")
            # print(f'tf trained {step} step: acc = {acc}')
    fold_step_l.append(np.argmax(gini_eval_l)*100)
    cv_gini_score_l.append(max(gini_eval_l))
    cv_gini_score += max(gini_eval_l) / nfold
    saver.restore(sess, base_path+'model.ckpt')
    sub['target'] += sess.run(y_pred_v, {tf_x: X_1})[:,1] / nfold
    sub_train['target'].iloc[test_index] = sess.run(y_pred_v, {tf_x: X_test})[:,1]
    # y_pred += sess.run(y_pred_v, {tf_x: X_1})[:,1] / nfold
    print("COST {} seconds\n".format(time.time()-t0))

print(f'CV gini score: {cv_gini_score:.6f}')
print(f'cv_gini_score_l: {cv_gini_score_l}')
print(f'average train steps: {sum(fold_step_l)/len(fold_step_l)}')





# %%
