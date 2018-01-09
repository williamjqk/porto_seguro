# %% this idea is from https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
base_path = '/home/tom/mywork/some_test/porto_seguro/input/'

model_name = 'porto_seguro_dae007'
data_path = '/home/tom/data/porto_seguro_dae'
model_path = os.path.join(data_path, model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)

import numpy as np
np.random.seed(20)
import pandas as pd
from PPMoney.core.data import HDFDataSet
from tensorflow import set_random_seed

from sklearn.model_selection import StratifiedKFold

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
import lightgbm as lgb

params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'max_bin': 255,
    'num_leaves': 31,
    'min_data_in_leaf': 1500,
    'feature_fraction': 0.7,
    'bagging_freq': 1,
    'bagging_fraction': 0.7,
    'lambda_l1': 1,
    'lambda_l2': 1,
}

n_rounds = 1400
test_cv = 5

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
def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True

def lgb_cv_scores_testset(gbdt, params, X_all, y_all, X_test, feval, cv=5, random_seed=0, early_stopping_rounds=100, verbose_eval=10, max_rounds=1e5):
    # def feval(preds, dtrain):
    #     y = dtrain.get_label()
    #     score = fscore(y,preds)
    #     fname = fscore.__name__
    #     return fname, score, True
    nrounds = int(max_rounds)
    kfold = cv
    skf = StratifiedKFold(n_splits=kfold, random_state=0)
    test_predict = np.zeros((X_test.shape[0], 1))
    test_predict_kfold = np.zeros((X_test.shape[0], cv))
    scores_l = []
    # ks_scores_l = []
    for i, (train_index, valid_index) in enumerate(skf.split(X_all, y_all)):
        print(' gbdt kfold: {}  of  {} : '.format(i+1, kfold))
        X_train, X_valid = X_all[train_index], X_all[valid_index]
        y_train, y_valid = y_all[train_index], y_all[valid_index]
        d_train = gbdt.Dataset(X_train, y_train)
        d_valid = gbdt.Dataset(X_valid, y_valid)
        eval_res = dict()
        # watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        gbdt_model = gbdt.train(params, d_train, nrounds, valid_sets=d_valid, early_stopping_rounds=early_stopping_rounds,
                              feval=feval, evals_result=eval_res, verbose_eval=verbose_eval)
        # test_predict += gbdt_model.predict(gbdt.DMatrix(X_test)) / kfold
        test_predict = gbdt_model.predict(X_test, num_iteration=gbdt_model.best_iteration)
        test_predict_kfold[:,i] = test_predict.reshape(-1,)
        eval_score = eval_res['valid_0']['gini'][gbdt_model.best_iteration]
        scores_l.append(eval_score)
        # scores_l.append(fscore(y_test, test_predict))
        # ks_scores_l.append(ks_score(y_test, test_predict))
        # break
    return scores_l, test_predict_kfold, gbdt_model

# d_train = lgb.Dataset(X_0, y_0)
# lgb_model = lgb.train(params, d_train, 22, valid_sets=d_train,
#                       feval=gini_lgb, verbose_eval=10)
test_scores_l, test_predict_kfold, gbdt_model = lgb_cv_scores_testset(lgb, params, X_0, y_0, X_1, gini_lgb, cv=test_cv,
                                                            max_rounds=2000, verbose_eval=100, early_stopping_rounds=200)

print(f'DATASET shapes: X_0.shape, X_1.shape: {X_0.shape, X_1.shape}')
print(f'gini-avg-score: {np.mean(test_scores_l):.6f}, cv-std: {np.std(test_scores_l):.6f}')
# dae007: gini-avg-score: 0.286470, cv-std: 0.004508,
# dae001: gini-avg-score: 0.287981, cv-std: 0.004854, PublicLB: 0.28380, PrivateLB: 0.29065 # NOTE: this result is the same as mjahrer's #1


sub['target'] = np.mean(test_predict_kfold, axis=1)
sub.describe()
sub.to_csv(os.path.join(model_path, 'test_'+model_name+'.csv.gz'), index=False, float_format='%.5f', compression='gzip')
# sub_train.to_csv(os.path.join(model_path, 'train_'+model_name+'.csv'), index=False, float_format='%.5f')
