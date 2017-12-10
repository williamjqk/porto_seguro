# %% this idea is from https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'

model_name = 'porto_seguro_dae00x1'
data_path = '/home/ljc/data/porto_seguro_dae'
model_path = os.path.join(data_path, model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)

import numpy as np
np.random.seed(20)
import pandas as pd

from tensorflow import set_random_seed

from sklearn.model_selection import StratifiedKFold

'''Data loading & preprocessing
'''

X_train_raw = pd.read_csv(base_path+'train.csv')
X_test_raw = pd.read_csv(base_path+'test.csv')

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

# %% 希望将1st使用的特征工程后的数据集用HDFDataSet存成h5
from PPMoney.core.data import HDFDataSet
dataset_tr = HDFDataSet(os.path.join(data_path, 'mjahrer_1st_train.dataset'), chunk_size=2048)
dataset_tr.add({'label': y_0, 'feature': X_0})
dataset_t = HDFDataSet(os.path.join(data_path, 'mjahrer_1st_test.dataset'), chunk_size=2048)
dataset_t.add({'feature': X_1})

# 从文件中读入dataset
dataset_load = HDFDataSet(os.path.join(data_path, 'mjahrer_1st_train.dataset'), chunk_size=2048)
dataset_load['feature']
dataset_load['label']
