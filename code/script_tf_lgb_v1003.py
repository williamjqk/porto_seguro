# %% XGBOOST and LGB from kaggle kernel https://www.kaggle.com/rshally/porto-xgb-lgb-kfold-lb-0-282¶
from datetime import datetime
from collections import Counter
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import gc

print('loading files...')

base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
train = pd.read_csv(base_path+'train.csv', na_values=-1)
test = pd.read_csv(base_path+'test.csv', na_values=-1)


train.isnull().sum()
test.isnull().sum()

print(train.shape, test.shape)
Counter(train.dtypes)


import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series(["thisisanewclass"
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
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


from QPhantom.core.preprocessing import AutoScaler
# from PPMoney.core.preprocessing import AutoScaler
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

# %% tf generate hidden features
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
print(f'X_tf.shape, y_tf.shape: {X_tf.shape, y_tf.shape}')

import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
in_dim = X_tf.shape[1]
tf_param = {
    # "layers": [4096,256,256,128],
    # "layers": [4096,2048,512,128],
    # "layers": [1024,512,256,128],
    "layers": [1024,512,256],
    "drop": 0.8,#0.2,#0.5, # in keras its drop, in tf its keep
    "noise_stddev": 0.2
}

shuffle_random_range = np.arange(X_tf.shape[0])
for i in range(6):
    np.random.shuffle(shuffle_random_range)

X_tf = X_tf[shuffle_random_range]
y_tf = y_tf[shuffle_random_range]


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


tf_x = tf.placeholder(tf.float32, [None, in_dim]) #  X_train.shape
tf_y = tf.placeholder(tf.int32, None)
fc0 = tf.layers.dense(tf_x, tf_param['layers'][0])
# dropout0 = tf.nn.dropout(fc0, tf_param['drop'])
# batch0 = tf.layers.batch_normalization(dropout0, tf_param['noise_stddev'])
fc1 = tf.layers.dense(fc0, tf_param['layers'][1], tf.nn.relu)
dropout1 = tf.nn.dropout(fc1, tf_param['drop'])
batch1 = tf.layers.batch_normalization(dropout1)
gs_noise1 = gaussian_noise_layer(batch1, tf_param['noise_stddev'])
fc2 = tf.layers.dense(gs_noise1, tf_param['layers'][2], tf.nn.relu)
dropout2 = tf.nn.dropout(fc2, tf_param['drop'])
batch2 = tf.layers.batch_normalization(dropout2)
gs_noise2 = gaussian_noise_layer(batch2, tf_param['noise_stddev'])
# fc3 = tf.layers.dense(gs_noise2, tf_param['layers'][3], tf.nn.relu)
# dropout3 = tf.nn.dropout(fc3, tf_param['drop'])
# batch3 = tf.layers.batch_normalization(dropout3)
# gs_noise3 = gaussian_noise_layer(batch3, tf_param['noise_stddev'])
output = tf.layers.dense(gs_noise2, 2)                     # output laye
# output = tf.layers.dense(gs_noise3, 2)                     # output laye

loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=output)           # compute cost
accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.squeeze(tf_y), predictions=tf.argmax(output, axis=1),)[1]
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

batch_size = 50#100
n_steps = X_tf.shape[0] // batch_size
n_steps *= 2#1

batch_gen = my_generator(X_tf, y_tf, batch_size=batch_size)

# ## TRAIN nn:
# for step in range(n_steps):
#     # train and net output
#     X_batch, y_batch = next(batch_gen)
#     _, acc, pred = sess.run([train_op, accuracy, output], {tf_x: X_batch, tf_y: y_batch})
#     if step % 100 == 0:
#         print(f'tf trained {step} step: acc = {acc}')



# custom objective function (similar to auc)
def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred) / gini(y, y)

def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True


fc0_new_0 = sess.run(fc0, {tf_x: X_0})
fc1_new_0 = sess.run(fc1, {tf_x: X_0})
# Xnew_0 = np.hstack((X_0, fc0_new_0, fc1_new_0))
# Xnew_0 = np.hstack((X_0, fc0_new_0))
Xnew_0 = X_0
Xnew_0.shape
y_new_0 = y_0
y_new_0.shape

fc0_new_1 = sess.run(fc0, {tf_x: X_1})
fc1_new_1 = sess.run(fc1, {tf_x: X_1})
# Xnew_1 = np.hstack((X_1, fc0_new_1, fc1_new_1))
# Xnew_1 = np.hstack((X_1, fc0_new_1))
Xnew_1 = X_1
Xnew_1.shape





X = Xnew_0#X_0
y = y_new_0#y_0
sub = test['id'].to_frame()
sub['target'] = 0

sub_train = train['id'].to_frame()
sub_train['target'] = 0

nrounds=10**6  # need to change to 2000
kfold = 5  # need to change to 5
seed = datetime.now().second + datetime.now().minute # 7
# np.random.seed(seed)
skf = StratifiedKFold(n_splits=kfold, random_state=seed)

# lgb
sub['target']=0
sub_train['target']=0


# params = {'metric': 'auc', 'learning_rate' : 0.01, 'max_depth':8, 'max_bin':10,  'objective': 'binary',
#           'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':5,  'min_data': 500}
# params = {'metric': 'auc', 'learning_rate' : 0.03, 'num_leaves': 64, 'boosting_type': 'gbdt',
#   'objective': 'binary', 'feature_fraction': 0.9,'bagging_fraction':0.8,'bagging_freq':3}
params = {'metric': 'auc', 'learning_rate' : 0.03, 'num_leaves': 64, 'boosting_type': 'gbdt',
  'objective': 'binary', 'feature_fraction': 0.9,'bagging_fraction':0.8,'bagging_freq':3,
  'lambda_l1': 2.0, 'lambda_l2': 2.0, }



eval_score_kfold = 0
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(' lgb kfold: {}  of  {} : '.format(i+1, kfold))
    X_train, X_eval = X[train_index], X[test_index]
    y_train, y_eval = y[train_index], y[test_index]
    eval_res = dict()
    lgb_model = lgb.train(params, lgb.Dataset(X_train, label=y_train), nrounds,
                  lgb.Dataset(X_eval, label=y_eval), verbose_eval=100,
                  feval=gini_lgb, evals_result=eval_res, early_stopping_rounds=100)
    eval_score_kfold += eval_res['valid_0']['gini'][lgb_model.best_iteration-1] / (kfold)
    sub['target'] += lgb_model.predict(Xnew_1, # test[features].values
                        num_iteration=lgb_model.best_iteration) / (kfold)
    sub_train['target'] += lgb_model.predict(Xnew_0, # train[features].values
                        num_iteration=lgb_model.best_iteration) / (kfold)
print(f'AVERAGE eval score: {eval_score_kfold}')
sub.to_csv(base_path+'test_sub_tf_lgb_scale_impute_v1003.csv', index=False, float_format='%.5f')
sub_train.to_csv(base_path+'train_sub_tf_lgb_scale_impute_v1003.csv', index=False, float_format='%.5f')
len(sub)

# eval_res = dict()
# eval_res['valid_0']['gini'][lgb_model.best_iteration-1]
# lgb_model = lgb.train(params, lgb.Dataset(X_train, label=y_train), nrounds,
#               lgb.Dataset(X_eval, label=y_eval), verbose_eval=100,
#               feval=gini_lgb, evals_result=eval_res, early_stopping_rounds=100)
# lgb_model.eval_valid(feval=gini_lgb)

gc.collect()
sub.head(2)
