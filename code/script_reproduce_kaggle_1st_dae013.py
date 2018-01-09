# %% this idea is from https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
base_path = '/home/tom/mywork/some_test/porto_seguro/input/'

model_name = 'porto_seguro_dae013'
data_path = '/home/tom/data/porto_seguro_dae'
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
    "learning_rate": 1e-4, #1e-3, # 3e-3,
    "minibatch_size": 128,
    "learning_rate_decay": 0.995,
    "keep_rate": 0.5,
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

optimizer = tf.train.RMSPropOptimizer(learning_rate = dae_params['learning_rate'])

# global_step = tf.Variable(0, trainable=False)
# initial_learning_rate = dae_params['learning_rate'] #初始学习率
# learning_rate = tf.train.exponential_decay(initial_learning_rate,
#                                            global_step=global_step,
#                                            decay_steps=10,decay_rate=0.9)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# add_global = global_step.assign_add(1)

# add_global = global_step.assign_add(1)
# with tf.control_denpendices([add_global]):
#     train_op = opt.minimise(loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops+[add_global]):
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

def batch_idx_generator(y_0, batch_size=64):
    # idx0 = np.arange(0, y_0.shape[0])
    # Counter(y_0)
    # counter1 = Counter(y_0)
    # counter1_l = sorted(counter1, key=counter1.get)
    # balance_d = [counter1_l[0], counter1[counter1_l[-1]]//counter1[counter1_l[0]]]
    # mask_0 = (y_0 == balance_d[0])
    # idx_ = np.tile(idx0[mask_0], balance_d[1])
    # print(idx_.shape, idx0.shape)
    # idx = np.hstack((idx0, idx_))

    idx = np.arange(y_0.shape[0])

    shuffle_random_range = np.arange(idx.shape[0])
    for i in range(6):
        np.random.shuffle(shuffle_random_range)
    idx = idx[shuffle_random_range]

    N = idx.shape[0]
    m = batch_size
    steps = (N-1)//m + 1
    pad_size = steps*m - N
    # Xpad = np.vstack(X, randomchoose(X, padsize))
    while True:
        for i in range(steps):
            # X_batch = X[(m*i):(m*(i+1)), :]
            idx_batch = idx[(m*i):(m*(i+1))]
            if pad_size > 0 and i == steps-1:
                rand_idx = np.random.choice(N, pad_size)
                # X_batch = np.vstack((X_batch, X[rand_idx]))
                idx_batch = np.hstack((idx_batch, idx[rand_idx]))
            yield idx_batch

# shuffle_random_range = np.arange(X_0.shape[0])
# for i in range(6):
#     np.random.shuffle(shuffle_random_range)
# X_0 = X_0[shuffle_random_range]

def predict_by_minibatch(X_input, batch_size):
    y_out_stack = np.zeros((X_input.shape[0],))
    i = 0
    while True:
        y_out_stack[i*batch_size:(i+1)*batch_size] += sess.run(y_pred_v, {tf_x: X_input[i*batch_size:(i+1)*batch_size,:]})[:,1]
        i += 1
        if i*batch_size >= X_input.shape[0]:
            break
    return y_out_stack

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
n_epochs = 150

cv_gini_score = 0
cv_gini_score_l = []
fold_step_l = []
tn_l = []
for i1, (train_index, test_index) in enumerate(skf.split(X_0,y_0)):
    t0 = time.time()
    # train_index, test_index = tmp_index[0]
    print(f"THE {i1}th fold, ", "TRAIN:", train_index.shape, "TEST:", test_index.shape)
    X_train_0, X_test = X_0[train_index], X_0[test_index]
    y_train_0, y_test = y_0[train_index], y_0[test_index]

    shuffle_random_range = np.arange(X_train_0.shape[0])
    for i in range(6):
        np.random.shuffle(shuffle_random_range)
    X_train_0 = X_train_0[shuffle_random_range]
    y_train_0 = y_train_0[shuffle_random_range]

    n_steps = (y_train_0.shape[0]-1) // batch_size + 1
    n_steps *= n_epochs#1

    # X_train,y_train = balance_train_data(X_train_0, y_train_0)
    # # X_train,y_train = balance_train_data(X_train_0, y_train_0)
    # # X_train,y_train = (X_train_0, y_train_0)

    # batch_gen = batch_generator(X_train, y_train, batch_size=batch_size)
    # batch_idx_gen = batch_idx_generator(y_train_0, batch_size=batch_size)

    batch_gen = batch_generator(X_train_0, y_train_0, batch_size=batch_size)


    # batch_queue.data_queue.qsize()

    print('define a batch generator')
    eval_ks_l = []
    gini_eval_l = []

    config = tf.ConfigProto(device_count={"CPU": 80},
                        inter_op_parallelism_threads=1,
                        intra_op_parallelism_threads=1)
    sess = tf.Session(config=config)                                                                # control training and others
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)     # initialize var in graph

    tn_l = []
    tn_l.append(time.time())
    for step in range(n_steps):
        # train and net output

        X_batch, y_batch = next(batch_gen)
        # tn_l.append(time.time());print(f"idx_batch, Time cost of {len(tn_l)}: {tn_l[-1] - tn_l[-2]}")
        # _, acc, pred = sess.run([train_op, accuracy, output], {tf_x: X_batch, tf_y: y_batch})
        # _, pred = sess.run([train_op, output], {tf_x: X_batch, tf_y: y_batch})
        #
        # X_batch = X_0[train_index][idx_batch]
        # y_batch = y_0[train_index][idx_batch]
        # tn_l.append(time.time());print(f"get batch, Time cost of {len(tn_l)}: {tn_l[-1] - tn_l[-2]}")
        result, loss_step, _ = sess.run([merge_op, loss, train_op],
                                {tf_x: X_batch,
                                 tf_y: y_batch,
                                 keep_rate: dae_params['keep_rate'],
                                 input_swap_noise: dae_params['input_swap_noise'],
                                 noise_std: dae_params['noise_std'],
                                 bn_phase: True})
        # tn_l.append(time.time());print(f"sess.run, Time cost of {len(tn_l)}: {tn_l[-1] - tn_l[-2]}")


        if step % 100 == 0:
            writer.add_summary(result, step)

            y_train_pred = predict_by_minibatch(X_train_0, 20000)
            tn_l.append(time.time());print(f"y_train_pred, Time cost of {len(tn_l)}: {tn_l[-1] - tn_l[-2]}")

            # y_train_pred = sess.run(y_pred_v, {tf_x: X_train_0})[:,1] # FIXME: X_train_0 is too large
            auc_train = roc_auc_score(y_train_0, y_train_pred)
            tn_l.append(time.time());print(f"auc_train, Time cost of {len(tn_l)}: {tn_l[-1] - tn_l[-2]}")
            # ks_train = ks_score(y_train_0, y_train_pred)
            gini_train = gini_mlp(y_train_pred, y_train_0)
            tn_l.append(time.time());print(f"gini_train, Time cost of {len(tn_l)}: {tn_l[-1] - tn_l[-2]}")
            # y_eval_pred = sess.run(y_pred_v, {tf_x: X_test})[:,1]
            y_eval_pred = predict_by_minibatch(X_test, 20000)
            tn_l.append(time.time());print(f"y_eval_pred, Time cost of {len(tn_l)}: {tn_l[-1] - tn_l[-2]}")
            auc_eval = roc_auc_score(y_test, y_eval_pred)
            tn_l.append(time.time());print(f"auc_eval, Time cost of {len(tn_l)}: {tn_l[-1] - tn_l[-2]}")
            # ks_eval = ks_score(y_test, y_eval_pred)
            gini_eval = gini_mlp(y_eval_pred, y_test)
            tn_l.append(time.time());print(f"gini_eval, Time cost of {len(tn_l)}: {tn_l[-1] - tn_l[-2]}")

            gini_eval_l.append(gini_eval)
            if np.argmax(gini_eval_l) == len(gini_eval_l)-1:
                saver.save(sess, os.path.join(model_path,'model.ckpt'))

            # if len(gini_eval_l) - np.argmax(gini_eval_l) > 50:
            #     print(f"Best step {np.argmax(gini_eval_l)*100} best EVAL gini score: {max(gini_eval_l):.6f}")
            #     print(f"{step} TRAIN auc: {auc_train:.6f}, gini: {gini_train:.6f}; EVAL auc: {auc_eval:.6f}, gini: {gini_eval:.6f}")
            #     break

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


# result, loss_step, _ = sess.run([merge_op, loss, train_op],
#                         {tf_x: X_0[train_index][idx_batch],
#                          tf_y: y_0[train_index][idx_batch],
#                          keep_rate: dae_params['keep_rate'],
#                          input_swap_noise: dae_params['input_swap_noise'],
#                          noise_std: dae_params['noise_std'],
#                          bn_phase: True})
#
# for i1 in range(10):
#     idx_batch = next(batch_idx_gen)
#     print(y_0[train_index][idx_batch])


print(f'CV gini score: {cv_gini_score:.6f}')
print(f'cv_gini_score_l: {cv_gini_score_l}')
print(f'average train steps: {sum(fold_step_l)/len(fold_step_l)}')





# %%
