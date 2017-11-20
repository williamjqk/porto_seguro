# %% XGBOOST and LGB from kaggle kernel https://www.kaggle.com/rshally/porto-xgb-lgb-kfold-lb-0-282¶
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
import numpy as np
np.random.seed(20)
import pandas as pd

from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.model_selection import StratifiedKFold

'''Data loading & preprocessing
'''

X_train = pd.read_csv(base_path+'train.csv')
X_test = pd.read_csv(base_path+'test.csv')

test_dat_0 = X_test
train_dat_0 = X_train

X_train, y_train = X_train.iloc[:,2:], X_train.target
X_test, test_id = X_test.iloc[:,1:], X_test.id

#OHE / some feature engineering adapted from the1owl kernel at:
#https://www.kaggle.com/the1owl/forza-baseline/code

#excluded columns based on snowdog's old school nn kernel at:
#https://www.kaggle.com/snowdog/old-school-nnet

X_train['negative_one_vals'] = np.sum((X_train==-1).values, axis=1)
X_test['negative_one_vals'] = np.sum((X_test==-1).values, axis=1)

to_drop = ['ps_car_11_cat', 'ps_ind_14', 'ps_car_11', 'ps_car_14', 'ps_ind_06_bin',
           'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin',
           'ps_ind_13_bin']

cols_use = [c for c in X_train.columns if (not c.startswith('ps_calc_'))
             & (not c in to_drop)]

X_train = X_train[cols_use]
X_test = X_test[cols_use]

one_hot = {c: list(X_train[c].unique()) for c in X_train.columns}

#note that this encodes the negative_one_vals column as well
for c in one_hot:
    if len(one_hot[c])>2 and len(one_hot[c]) < 105:
        for val in one_hot[c]:
            newcol = c + '_oh_' + str(val)
            X_train[newcol] = (X_train[c].values == val).astype(np.int)
            X_test[newcol] = (X_test[c].values == val).astype(np.int)
        X_train.drop(labels=[c], axis=1, inplace=True)
        X_test.drop(labels=[c], axis=1, inplace=True)

X_train = X_train.replace(-1, np.NaN)  # Get rid of -1 while computing interaction col
X_test = X_test.replace(-1, np.NaN)

X_train['ps_car_13_x_ps_reg_03'] = X_train['ps_car_13'] * X_train['ps_reg_03']
X_test['ps_car_13_x_ps_reg_03'] = X_test['ps_car_13'] * X_test['ps_reg_03']

X_train = X_train.fillna(-1)
X_test = X_test.fillna(-1)

X_0 = X_train.values
y_0 = y_train.values

X_1 = X_test.values


# %% scale
from PPMoney.core.preprocessing import AutoScaler
from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
scaler = AutoScaler(threshold=20.0)


scaler.fit(np.vstack((X_0, X_1)))
X_0 = scaler.transform(X_0)
X_1 = scaler.transform(X_1)

# %%
sub = test_dat_0['id'].to_frame()
sub['target'] = 0

sub_train = train_dat_0['id'].to_frame()
sub_train['target'] = 0

# %%
import numpy
import pandas
import tensorflow as tf
tf.reset_default_graph()
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics

# -------- use tf BEGIN ---------
# fix random seed for reproducibility
from collections import Counter
from datetime import datetime
seed = datetime.now().second + datetime.now().minute
np.random.seed(seed)
# from sklearn import cross_validation
from sklearn.model_selection import StratifiedKFold
nfold = 5
skf = StratifiedKFold(n_splits=nfold, random_state=seed)
# skf = StratifiedKFold()
print(skf)
print(Counter(y_0))


tf_param = {
    # # "layers": [4096,256,256,128],
    # # "layers": [4096,2048,512,128],
    # # "layers": [1024,512,256,128],
    # # "layers": [2048,1024,256,128],
    # # "layers": [256,256,256,128],
    # # "layers": [2048,1024,256,128],
    # # "layers": [4096,2048,512,128],
    # # "layers": [700,256,256,128],
    # # "layers": [1024,1024,256,128],
    # # "layers": [1024,256,256,128],
    # # "layers": [1200,400,100],
    # "layers": [1500,800,200],
    # "drop": 0.5,#0.8,#0.2,#0.5, # in keras its drop, in tf its keep
    # "noise_stddev": 0.2#0.2
    "layers": [1500,800,200], "drop": 0.5, "noise_stddev": 0.2
    # "layers": [4000,2000,500,100], "drop": 0.5, "noise_stddev": 0.2 # 特别大的nn增加的capacity对数据空间的流形作用不大，而且增加了训练难度
}

def standard_layer(input_layer, n_nodes, std, keep_rate, phase):
    # layer1 = tf.layers.dense(input_layer, n_nodes, tf.nn.relu)
    # layer2 = tf.layers.batch_normalization(layer1, training=phase)
    # layer3 = tf.nn.dropout(layer2, keep_rate)
    # # layer2 = tf.nn.dropout(layer1, keep_rate)
    # # layer3 = tf.layers.batch_normalization(layer2, training=phase)
    # layer4 = gaussian_noise_layer(layer3, std)

    layer1 = tf.layers.dense(input_layer, n_nodes)
    layer1 = tf.layers.batch_normalization(layer1, training=phase)
    layer1 = tf.nn.relu(layer1)
    layer1 = gaussian_noise_layer(layer1, std)
    layer1 = tf.nn.dropout(layer1, keep_rate)
    return layer1
def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

in_dim = X_0.shape[1]
keep_rate = tf.placeholder_with_default(1.0, shape=())
noise_std = tf.placeholder_with_default(0.0, shape=())
phase = tf.placeholder_with_default(True, shape=()) # default: False

tf_x = tf.placeholder(tf.float32, [None, in_dim]) #  X_train.shape
tf_y = tf.placeholder(tf.int32, None)

hidden = tf_x
for i in range(len(tf_param['layers'])):
    hidden = standard_layer(hidden, tf_param['layers'][i], \
                            noise_std, keep_rate, phase)

output = tf.layers.dense(hidden, 2)

y_pred_v = tf.nn.softmax(output)

loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=output)           # compute cost
# accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
#     labels=tf.squeeze(tf_y), predictions=tf.argmax(output, axis=1),)[1]
# loss = tf.losses.log_loss(labels=tf_y, predictions=output)           # compute cost

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

sess = tf.Session()                                                                 # control training and others
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)     # initialize var in graph


def my_generator(X, y, batch_size=50):
    n_steps = X.shape[0] // batch_size
    while True:
        for step in range(n_steps):
            X_batch = X[(batch_size*step):(batch_size*(step+1)), :]
            y_batch = y[(batch_size*step):(batch_size*(step+1))]
            yield (X_batch, y_batch)

def shuffle_train_data(X_0, y_0):

    X_tf = X_0
    y_tf = y_0

    shuffle_random_range = np.arange(X_tf.shape[0])
    for i in range(6):
        np.random.shuffle(shuffle_random_range)

    X_tf = X_tf[shuffle_random_range]
    y_tf = y_tf[shuffle_random_range]

    print(f'X_tf.shape, y_tf.shape: {X_tf.shape, y_tf.shape}')
    return X_tf, y_tf

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

    X_tf, y_tf = shuffle_train_data(X_tf, y_tf)

    return X_tf, y_tf

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

batch_size = 80
n_steps = X_0.shape[0] // batch_size
n_steps *= 1000#1

cv_gini_score = 0
cv_gini_score_l = []
fold_step_l = []
for i1, (train_index, test_index) in enumerate(skf.split(X_0,y_0)):
    t0 = time.time()
    # train_index, test_index = tmp_index[0]
    print(f"THE {i1}th fold, ", "TRAIN:", train_index.shape, "EVAL:", test_index.shape)
    X_train_0, X_test = X_0[train_index], X_0[test_index]
    y_train_0, y_test = y_0[train_index], y_0[test_index]

    X_train,y_train = balance_train_data(X_train_0, y_train_0)
    # X_train,y_train = shuffle_train_data(X_train_0, y_train_0)
    # X_train,y_train = (X_train_0, y_train_0)

    batch_gen = my_generator(X_train, y_train, batch_size=batch_size)
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
        _, pred = sess.run([train_op, output], {tf_x: X_batch, tf_y: y_batch, \
                            noise_std:tf_param['noise_stddev'], keep_rate:tf_param['drop'], \
                            phase:True})
        if step % 100 == 0:
            y_train_pred = predict_by_minibatch(X_train_0, batch_size=20000)
            # y_train_pred = sess.run(y_pred_v, {tf_x: X_train_0})[:,1]
            auc_train = roc_auc_score(y_train_0, y_train_pred)
            # ks_train = ks_score(y_train_0, y_train_pred)
            gini_train = gini_mlp(y_train_pred, y_train_0)

            y_eval_pred = predict_by_minibatch(X_test, batch_size=20000)
            # y_eval_pred = sess.run(y_pred_v, {tf_x: X_test})[:,1]
            auc_eval = roc_auc_score(y_test, y_eval_pred)
            # ks_eval = ks_score(y_test, y_eval_pred)
            gini_eval = gini_mlp(y_eval_pred, y_test)

            gini_eval_l.append(gini_eval)
            if np.argmax(gini_eval_l) == len(gini_eval_l)-1:
                saver.save(sess, base_path+'model.ckpt')

            if len(gini_eval_l) - np.argmax(gini_eval_l) > 50:#50
                print(f"Best step {np.argmax(gini_eval_l)*100} best EVAL gini score: {max(gini_eval_l):.6f}")
                print(f"{step} TRAIN auc: {auc_train:.6f}, gini: {gini_train:.6f}; EVAL auc: {auc_eval:.6f}, gini: {gini_eval:.6f}")
                break

            if step % 1000 == 0:
                print(f"{step} TRAIN auc: {auc_train:.6f}, gini: {gini_train:.6f}; EVAL auc: {auc_eval:.6f}, gini: {gini_eval:.6f}")
            # print(f'tf trained {step} step: acc = {acc}')
    fold_step_l.append(np.argmax(gini_eval_l)*100)
    cv_gini_score_l.append(max(gini_eval_l))
    cv_gini_score += max(gini_eval_l) / nfold
    saver.restore(sess, base_path+'model.ckpt')


    sub['target'] += predict_by_minibatch(X_1, batch_size=20000) / nfold
    sub_train['target'].iloc[test_index] = sess.run(y_pred_v, {tf_x: X_test})[:,1]
    # y_pred += sess.run(y_pred_v, {tf_x: X_1})[:,1] / nfold
    print("COST {} seconds\n".format(time.time()-t0))

print(f'CV gini score: {cv_gini_score:.6f}, score var:{np.var(cv_gini_score_l)}')
print(f'cv_gini_score_l: {cv_gini_score_l}')
print(f'average train steps: {sum(fold_step_l)/len(fold_step_l)}')

# 保证输出结果0～1之间，提交结果有效
from sklearn.preprocessing import MinMaxScaler
mmscaler = MinMaxScaler((0,1))
sub['target'] = mmscaler.fit_transform(sub['target'].values[:,np.newaxis])

# %%
sub.to_csv(base_path+'test_dnn_camnugent_fe_v2006.csv', index=False, float_format='%.5f')
sub_train.to_csv(base_path+'train_dnn_camnugent_fe_v2006.csv', index=False, float_format='%.5f')

# -------- use tf END ---------
print(f'CV gini score: {cv_gini_score:.6f}, score var:{np.var(cv_gini_score_l)}')
print(f'cv_gini_score_l: {cv_gini_score_l}')
print(f'average train steps: {sum(fold_step_l)/len(fold_step_l)}')

# CV gini score: 0.272588, score var:1.528406982340506e-05
# "layers": [1500,800,200], "drop": 0.5, "noise_stddev": 0.2

gc.collect()
sub.head(2)
