import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
base_path = '/home/tom/mywork/some_test/porto_seguro/input/'
import numpy as np
np.random.seed(20)
import pandas as pd

from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.model_selection import StratifiedKFold

'''Data loading & preprocessing
'''

X_train_0 = pd.read_csv(base_path+'train.csv')
X_test_0 = pd.read_csv(base_path+'test.csv')

X_train, y_train = X_train_0.iloc[:,2:], X_train_0.target
X_test, test_id = X_test_0.iloc[:,1:], X_test_0.id

# %%
all_features = X_train.columns.tolist()

categorical_features = [x for x in all_features if x[-3:]=='cat']
X_train[categorical_features].isnull().sum() # 没有缺失值，缺失值应该是补-1了


# X_train[categorical_features[0]]*X_train[categorical_features[1]]
# X_train[categorical_features[0]] - X_train[categorical_features[0]].min()
# X_train[categorical_features[0]]*100+X_train[categorical_features[1]]

from sklearn.preprocessing import LabelEncoder
# new_feature = X_train[categorical_features[0]]*100+X_train[categorical_features[1]]


X_cross_cates = X_train_0['id'].to_frame()
n_cate_cols = len(categorical_features)
for i1 in range(n_cate_cols):
    col1 = categorical_features[i1]
    if i1 == n_cate_cols - 2:
        break
    for i2 in range(i1+1,n_cate_cols):
        col2 = categorical_features[i2]
        new_feature = X_train[col1]*100+X_train[col2]
        new_feature = LabelEncoder().fit_transform(new_feature)
        X_cross_cates[f'{col1}_X_{col2}'] = new_feature

X_cross_cates.drop(labels=['id'], axis=1, inplace=True)
X_train = pd.concat([X_train, X_cross_cates], axis=1)

# %% get feature importances from simple xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import *
import gc
import warnings
import xgboost as xgb

def ginic(actual, pred):
    actual = np.asarray(actual)
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n

def gini_normalized(a, p):
    if p.ndim == 2:
        p = p[:,1]
    return ginic(a, p) / ginic(a, a)


def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score

params = {'eta': 0.025, 'max_depth': 4,
          'subsample': 0.9, 'colsample_bytree': 0.7,
          'colsample_bylevel':0.7,
            'min_child_weight':100,
            'alpha':4,
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 99, 'silent': True}
x1, x2, y1, y2 = train_test_split(X_train, y_train, test_size=0.25, random_state=99)



watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 5000,  watchlist, feval=gini_xgb, maximize=True,
                  verbose_eval=100, early_stopping_rounds=70)

feature_scores = model.get_score(importance_type='gain')
feature_l = sorted(feature_scores, key=feature_scores.get)[::-1]
for x1 in feature_l:
    print(f'{x1}: {feature_scores[x1]}')








# %%
X_train['negative_one_vals'] = np.sum((X_train==-1).values, axis=1)
X_test['negative_one_vals'] = np.sum((X_test==-1).values, axis=1)


cols_use = [c for c in X_train.columns if (not c.startswith('ps_calc_'))
             & (not c in to_drop)]
