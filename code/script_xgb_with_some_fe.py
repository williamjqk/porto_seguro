# %% FE of kueipo
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import *
import gc
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
#######################################

# Thanks Pascal and the1owl

# Pascal's Recovery https://www.kaggle.com/pnagel/reconstruction-of-ps-reg-03
# Froza's Baseline https://www.kaggle.com/the1owl/forza-baseline

# single XGB LB 0.285 will release soon.

#######################################

#### Load Data
train = pd.read_csv(base_path + 'train.csv')
test = pd.read_csv(base_path + 'test.csv')

###
y = train['target'].values
testid= test['id'].values

train.drop(['id','target'],axis=1,inplace=True)
test.drop(['id'],axis=1,inplace=True)

### Drop calc
unwanted = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(unwanted, axis=1)
test = test.drop(unwanted, axis=1)

### Great Recovery from Pascal's materpiece
### Great Recovery from Pascal's materpiece
### Great Recovery from Pascal's materpiece
### Great Recovery from Pascal's materpiece
### Great Recovery from Pascal's materpiece

def recon(reg):
    integer = int(np.round((40*reg)**2))
    for a in range(32):
        if (integer - a) % 31 == 0:
            A = a
    M = (integer - A)//31
    return A, M
train['ps_reg_A'] = train['ps_reg_03'].apply(lambda x: recon(x)[0])
train['ps_reg_M'] = train['ps_reg_03'].apply(lambda x: recon(x)[1])
train['ps_reg_A'].replace(19,-1, inplace=True)
train['ps_reg_M'].replace(51,-1, inplace=True)
test['ps_reg_A'] = test['ps_reg_03'].apply(lambda x: recon(x)[0])
test['ps_reg_M'] = test['ps_reg_03'].apply(lambda x: recon(x)[1])
test['ps_reg_A'].replace(19,-1, inplace=True)
test['ps_reg_M'].replace(51,-1, inplace=True)


### Froza's baseline
### Froza's baseline
### Froza's baseline
### Froza's baseline

d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
d_skew = train.skew(axis=0)
one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id','target']}

def transform_df(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id','target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
    for c in dcol:
        if '_bin' not in c: #standard arithmetic
            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)

    for c in one_hot:
        if len(one_hot[c])>2 and len(one_hot[c]) < 7:
            for val in one_hot[c]:
                df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
    return df

def multi_transform(df):
    print('Init Shape: ', df.shape)
    p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    p.close(); p.join()
    print('After Shape: ', df.shape)
    return df

train_kueipo = multi_transform(train)
test_kueipo = multi_transform(test)

# %% camnugent's FE
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
from sklearn.preprocessing import StandardScaler
import gc

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
test_dat = pd.read_csv(base_path + 'test.csv')
train_dat = pd.read_csv(base_path + 'train.csv')
submission = pd.read_csv(base_path + 'sample_submission.csv')

train_y = train_dat['target']
train_x = train_dat.drop(['target', 'id'], axis = 1)
test_dat_0 = test_dat
test_dat = test_dat.drop(['id'], axis = 1)

merged_dat = pd.concat([train_x, test_dat],axis=0)

#change data to float32
for c, dtype in zip(merged_dat.columns, merged_dat.dtypes):
    if dtype == np.float64:
        merged_dat[c] = merged_dat[c].astype(np.float32)

#one hot encode the categoricals
cat_features = [col for col in merged_dat.columns if col.endswith('cat')]
for column in cat_features:
    temp=pd.get_dummies(pd.Series(merged_dat[column]))
    merged_dat=pd.concat([merged_dat,temp],axis=1)
    merged_dat=merged_dat.drop([column],axis=1)

#standardize the scale of the numericals
numeric_features = [col for col in merged_dat.columns if '_calc_' in  str(col)]
numeric_features = [col for col in numeric_features if '_bin' not in str(col)]

# scaler = StandardScaler()
# scaled_numerics = scaler.fit_transform(merged_dat[numeric_features])
from PPMoney.core.preprocessing import AutoScaler
scaler = AutoScaler(threshold=20.0)
scaler.fit(merged_dat[numeric_features])
scaled_numerics = scaler.transform(merged_dat[numeric_features])
scaled_num_df = pd.DataFrame(scaled_numerics, columns =numeric_features )


merged_dat = merged_dat.drop(numeric_features, axis=1)

merged_dat = np.concatenate((merged_dat.values,scaled_num_df), axis = 1)

train_x = merged_dat[:train_x.shape[0]]
test_dat = merged_dat[train_x.shape[0]:]

train_camnugent = train_x
# y_0 = train_y
test_camnugent = test_dat


# %%
sub = test_dat_0['id'].to_frame()
sub['target'] = 0

sub_train = train_dat['id'].to_frame()
sub_train['target'] = 0
type(train_kueipo)
type(train_camnugent)
# %%
X = np.hstack((train_kueipo.values,train_camnugent))
y = train_y
X_1 = np.hstack((test_kueipo.values,test_camnugent))
print(X.shape, X_1.shape)

# params = {'metric': 'auc', 'learning_rate' : 0.01, 'max_depth':8, 'max_bin':10,  'objective': 'binary',
#           'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':5,  'min_data': 500}
params = {'metric': 'auc', 'learning_rate' : 0.02, 'num_leaves': 64, 'boosting_type': 'gbdt',
  'objective': 'binary', 'feature_fraction': 0.9,'bagging_fraction':0.8,'bagging_freq':3,
  'lambda_l1': 2.0, 'lambda_l2': 2.0, }


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

nrounds=10**6  # need to change to 2000
kfold = 4  # need to change to 5
skf = StratifiedKFold(n_splits=kfold, random_state=1)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(' lgb kfold: {}  of  {} : '.format(i+1, kfold))
    X_train, X_eval = X[train_index], X[test_index]
    y_train, y_eval = y[train_index], y[test_index]
    lgb_model = lgb.train(params, lgb.Dataset(X_train, label=y_train), nrounds,
                  lgb.Dataset(X_eval, label=y_eval), verbose_eval=100,
                  feval=gini_lgb, early_stopping_rounds=100)
    sub['target'] += lgb_model.predict(X_1,
                        num_iteration=lgb_model.best_iteration) / (kfold)
    train_pred = lgb_model.predict(X_eval,
                        num_iteration=lgb_model.best_iteration)
    sub_train['target'].iloc[test_index] = train_pred

# %%
sub.to_csv(base_path+'test_xgb_with_some_fe.csv', index=False, float_format='%.5f')
sub_train.to_csv(base_path+'train_xgb_with_some_fe.csv', index=False, float_format='%.5f')

gc.collect()
sub.head(2)
