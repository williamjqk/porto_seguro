# %% XGBOOST and LGB from kaggle kernel https://www.kaggle.com/rshally/porto-xgb-lgb-kfold-lb-0-282¶
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from datetime import datetime
from collections import Counter
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import gc

print('loading files...')

base_path = '/home/tom/mywork/some_test/porto_seguro/input/'
# train = pd.read_csv(base_path+'train.csv', na_values=-1)
# test = pd.read_csv(base_path+'test.csv', na_values=-1)
train = pd.read_csv(base_path+'train_p.csv', na_values=-1)
test = pd.read_csv(base_path+'test_p.csv', na_values=-1)


train.isnull().sum()
test.isnull().sum()

print(train.shape, test.shape)
Counter(train.dtypes)


import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin
# class DataFrameImputer(TransformerMixin):
#     def fit(self, X, y=None):
#         self.fill = pd.Series(["thisisanewclass"
#             if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
#             index=X.columns)
#         return self
#     def transform(self, X, y=None):
#         return X.fillna(self.fill)
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series(["NULL"
            if X[c].dtype == np.dtype('O') else X[c].mean() if X[c].nunique() > 3 else X[c].mean() if X[c].count()==0 else -1.0 for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)
X = train.drop(['id', 'target'], axis=1)
features = X.columns
X = X.values
y = train['target'].values
sub=test['id'].to_frame()
sub['target']=0
sub_train = train['id'].to_frame()
sub_train['target']=0

X_tr = train[features].astype(float)
X_t = test[features].astype(float)
X_all = pd.concat([X_tr,X_t], axis=0)
# X_all.dtypes
print(f'shapes of X_tr, X_t, X_all: {X_tr.shape}, {X_t.shape}, {X_all.shape}')

my_imputer = DataFrameImputer()
my_imputer.fit(X_all)
X_tr_imputed = my_imputer.transform(X_tr)


# from QPhantom.core.preprocessing import AutoScaler
from PPMoney.core.preprocessing import AutoScaler
scaler = AutoScaler(threshold=20.0)
scaler.fit(X_all.as_matrix())

X_0 = X_tr.as_matrix()
X_0 = scaler.transform(X_0)
X_0 = np.nan_to_num(X_0)
y_0 = train['target'].values
print(f'X_0.shape, y_0.shape: {X_0.shape, y_0.shape}')
# np.isnan(X_0).sum()
X_1 = X_t.as_matrix()
X_1 = scaler.transform(X_1)
X_1 = np.nan_to_num(X_1)
print(f'X_1.shape: {X_1.shape}')

# %%
sub = test['id'].to_frame()
sub['target'] = 0

sub_train = train['id'].to_frame()
sub_train['target'] = 0

# %%
import numpy
import pandas
import tensorflow as tf
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics
tf.reset_default_graph()

# -------- use tf BEGIN ---------
# fix random seed for reproducibility
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
    # "layers": [4096,256,256,128],
    # "layers": [4096,2048,512,128],
    # "layers": [1024,512,256,128],
    # "layers": [2048,1024,256,128],
    # "layers": [256,256,256,128],
    # "layers": [2048,1024,256,128],
    # "layers": [4096,2048,512,128],
    # "layers": [700,256,256,128],
    # "layers": [1024,1024,256,128],
    "layers": [600,256,256,128],
    # "semi_layers": [256,256,128],
    # "semi_layers": [512,256,128],
    "semi_layers": [256,256,128],
    "drop": 0.8,#0.2,#0.5, # in keras its drop, in tf its keep
    "noise_stddev": 0.2
}



def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

def standard_layer(input_layer, n_nodes, std, keep_rate):
    layer1 = tf.layers.dense(input_layer, n_nodes, tf.nn.relu)
    layer2 = tf.nn.dropout(layer1, keep_rate)
    layer3 = tf.layers.batch_normalization(layer2)
    layer4 = gaussian_noise_layer(layer3, std)
    return layer4

def semi_layers(input_layer, tf_param):
    tmp_layer = tf.nn.dropout(input_layer, tf_param['drop'])
    tmp_layer = tf.layers.batch_normalization(tmp_layer)
    tmp_layer = gaussian_noise_layer(tmp_layer, tf_param['noise_stddev'])
    for i in range(len(tf_param['semi_layers'])):
        tmp_layer = standard_layer(tmp_layer, tf_param['semi_layers'][i], \
                            tf_param['noise_stddev'], tf_param['drop'])
    return tmp_layer

def decoder_encoder(input_layer, tf_param):
    # encoder
    en0 = tf.layers.dense(input_layer, tf_param['semi_layers'][0], tf.nn.relu)
    en1 = tf.layers.dense(en0, tf_param['semi_layers'][1], tf.nn.relu)
    encoded = tf.layers.dense(en1, tf_param['semi_layers'][2])
    # encoded = tf.layers.dense(en2, tf_param['layers'][3])

    # decoder
    # de0 = tf.layers.dense(encoded, tf_param['layers'][3], tf.nn.tanh)
    de1 = tf.layers.dense(encoded, tf_param['semi_layers'][2], tf.nn.relu)
    de2 = tf.layers.dense(de1, tf_param['semi_layers'][1], tf.nn.relu)
    decoded = tf.layers.dense(de2, tf_param['semi_layers'][0], tf.nn.relu)
    return decoded

def dense_reuse(x, n_nodes, layer_name='dense1', reuse=True):
    with tf.variable_scope(layer_name) as s:
        output = tf.layers.dense(tf_x, n_nodes, reuse=reuse,  scope = s)
    # output = tf.contrib.layers.convolution2d(inputs = x, num_outputs = 10, kernel_size = [3, 3],
                #    stride = [1, 1], padding = 'VALID', reuse = reuse, scope = s)
    return output


in_dim = X_0.shape[1]
tf_x = tf.placeholder(tf.float32, [None, in_dim]) #  X_train.shape
tf_y = tf.placeholder(tf.int32, None)
tf_x_semi = tf.placeholder(tf.float32, [None, in_dim]) #  X_train.shape
tf_y_semi = tf.placeholder(tf.int32, None)

with tf.variable_scope('dense1'):
    fc0 = tf.layers.dense(tf_x, tf_param['layers'][0], reuse=None)
    fc0_semi = tf.layers.dense(tf_x_semi, tf_param['layers'][0], reuse=True)
# fc0 = dense_reuse(tf_x, tf_param['layers'][0], layer_name='dense1', reuse=None)
# fc0_semi = dense_reuse(tf_x_semi, tf_param['layers'][0], layer_name='dense1', reuse=True)

# semi_out = semi_layers(fc0_semi, tf_param)
# semi_out = tf.layers.dense(semi_out, 2)
# loss_semi = tf.losses.sparse_softmax_cross_entropy(labels=tf_y_semi, logits=semi_out)

# semi_out = semi_layers(fc0_semi, tf_param)
# semi_out = tf.layers.dense(semi_out, in_dim)
semi_out = decoder_encoder(fc0_semi, tf_param)
semi_out = tf.layers.dense(semi_out, tf_param['layers'][0], tf.nn.relu)
semi_out = tf.layers.dense(semi_out, in_dim)
loss_semi = tf.losses.mean_squared_error(labels=tf_x_semi, predictions=semi_out)

dropout0 = tf.nn.dropout(fc0, tf_param['drop'])
batch0 = tf.layers.batch_normalization(dropout0)
gs_noise0 = gaussian_noise_layer(batch0, tf_param['noise_stddev'])

fc1 = tf.layers.dense(gs_noise0, tf_param['layers'][1], tf.nn.relu)
dropout1 = tf.nn.dropout(fc1, tf_param['drop'])
batch1 = tf.layers.batch_normalization(dropout1)
gs_noise1 = gaussian_noise_layer(batch1, tf_param['noise_stddev'])
fc2 = tf.layers.dense(gs_noise1, tf_param['layers'][2], tf.nn.relu)
dropout2 = tf.nn.dropout(fc2, tf_param['drop'])
batch2 = tf.layers.batch_normalization(dropout2)
gs_noise2 = gaussian_noise_layer(batch2, tf_param['noise_stddev'])
fc3 = tf.layers.dense(gs_noise2, tf_param['layers'][3], tf.nn.relu)
dropout3 = tf.nn.dropout(fc3, tf_param['drop'])
batch3 = tf.layers.batch_normalization(dropout3)
gs_noise3 = gaussian_noise_layer(batch3, tf_param['noise_stddev'])
# output = tf.layers.dense(gs_noise3, 1, tf.nn.sigmoid)                     # output laye
output = tf.layers.dense(gs_noise3, 2)                     # output laye

y_pred_v = tf.nn.softmax(output)

loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=output)           # compute cost
# accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
#     labels=tf.squeeze(tf_y), predictions=tf.argmax(output, axis=1),)[1]
# loss = tf.losses.log_loss(labels=tf_y, predictions=output)           # compute cost

# loss += 0.2 * loss_semi

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
optimizer = tf.train.AdamOptimizer() # learning_rate=0.0001
train_op = optimizer.minimize(loss)

optimizer_semi = tf.train.AdamOptimizer() # learning_rate=0.0001
train_op_semi = optimizer_semi.minimize(loss_semi)

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

batch_size = 100
batch_size_semi = 100
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
    # X_train,y_train = (X_train_0, y_train_0)

    batch_gen = my_generator(X_train, y_train, batch_size=batch_size)
    batch_gen_semi = my_generator(X_1, np.ones((X_1.shape[0],)), batch_size=batch_size_semi)
    print('define a batch generator')
    eval_ks_l = []
    gini_eval_l = []

    sess = tf.Session()                                                                 # control training and others
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)     # initialize var in graph

    for step in range(n_steps):
        # train and net output
        X_batch, y_batch = next(batch_gen)
        X_batch_semi, y_batch_semi = next(batch_gen_semi)

        # _, acc, pred = sess.run([train_op, accuracy, output], {tf_x: X_batch, tf_y: y_batch})
        # _, pred = sess.run([train_op, output], {tf_x: X_batch, tf_y: y_batch, \
        #                                         tf_x_semi:X_batch_semi, tf_y_semi:y_batch_semi})
        for i in range(30):
            _ = sess.run(train_op_semi, {tf_x_semi:X_batch_semi})

        _, pred = sess.run([train_op, output], {tf_x: X_batch, tf_y: y_batch})


        if step % 100 == 0:
            y_train_pred = sess.run(y_pred_v, {tf_x: X_train_0})[:,1]
            auc_train = roc_auc_score(y_train_0, y_train_pred)
            # ks_train = ks_score(y_train_0, y_train_pred)
            gini_train = gini_mlp(y_train_pred, y_train_0)
            y_eval_pred = sess.run(y_pred_v, {tf_x: X_test})[:,1]
            auc_eval = roc_auc_score(y_test, y_eval_pred)
            # ks_eval = ks_score(y_test, y_eval_pred)
            gini_eval = gini_mlp(y_eval_pred, y_test)

            gini_eval_l.append(gini_eval)
            if np.argmax(gini_eval_l) == len(gini_eval_l)-1:
                saver.save(sess, base_path+'model.ckpt')

            if len(gini_eval_l) - np.argmax(gini_eval_l) > 50:# 50
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
    sub['target'] += sess.run(y_pred_v, {tf_x: X_1})[:,1] / nfold
    sub_train['target'].iloc[test_index] = sess.run(y_pred_v, {tf_x: X_test})[:,1]
    # y_pred += sess.run(y_pred_v, {tf_x: X_1})[:,1] / nfold
    print("COST {} seconds\n".format(time.time()-t0))

print(f'CV gini score: {cv_gini_score:.6f}; score var: {np.var(cv_gini_score_l):.3e}')
print(f'cv_gini_score_l: {cv_gini_score_l}')
print(f'average train steps: {sum(fold_step_l)/len(fold_step_l)}')


sub.to_csv(base_path+'test_sub_v2001.csv', index=False, float_format='%.5f')
sub_train.to_csv(base_path+'train_sub_v2001.csv', index=False, float_format='%.5f')

# -------- use tf END ---------


gc.collect()
sub.head(2)
