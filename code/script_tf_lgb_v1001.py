# %% XGBOOST and LGB from kaggle kernel https://www.kaggle.com/rshally/porto-xgb-lgb-kfold-lb-0-282¶
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import gc

print('loading files...')

base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
train = pd.read_csv(base_path+'train_p.csv', na_values=-1)
test = pd.read_csv(base_path+'test_p.csv', na_values=-1)

train.shape
train.isnull().sum()
test.isnull().sum()

print(train.shape, test.shape)


import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        # self.fill = pd.Series([X[c].value_counts().index[0]
        #     if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
        #     index=X.columns)
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


X = X_0
y = y_0
sub = test['id'].to_frame()
sub['target'] = 0

sub_train = train['id'].to_frame()
sub_train['target'] = 0

nrounds=10**6  # need to change to 2000
kfold = 5  # need to change to 5
skf = StratifiedKFold(n_splits=kfold, random_state=0)

# lgb
sub['target']=0
sub_train['target']=0

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    # 'metric': {'auc'}, # ,'binary_logloss'
    # 'scale_pos_weight': 8,
    'num_leaves': 64,
    'learning_rate': 0.03,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 3,
    'verbose': 0
}

params = {'metric': 'auc', 'learning_rate' : 0.03, 'num_leaves': 64, 'boosting_type': 'gbdt',
  'objective': 'binary', 'feature_fraction': 0.9,'bagging_fraction':0.8,'bagging_freq':3}
# params = {'metric': 'auc', 'learning_rate' : 0.01, 'max_depth':8, 'max_bin':10,  'objective': 'binary',
#           'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':5,  'min_data': 500}


skf = StratifiedKFold(n_splits=kfold, random_state=1)
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
    sub['target'] += lgb_model.predict(X_1, # test[features].values
                        num_iteration=lgb_model.best_iteration) / (kfold)
    sub_train['target'] += lgb_model.predict(X_0, # train[features].values
                        num_iteration=lgb_model.best_iteration) / (kfold)
print(f'AVERAGE eval score: {eval_score_kfold}')
sub.to_csv(base_path+'test_sub_lgb_scale_impute.csv', index=False, float_format='%.5f')
sub_train.to_csv(base_path+'train_sub_lgb_scale_impute.csv', index=False, float_format='%.5f')

# eval_res = dict()
# eval_res['valid_0']['gini'][lgb_model.best_iteration-1]
# lgb_model = lgb.train(params, lgb.Dataset(X_train, label=y_train), nrounds,
#               lgb.Dataset(X_eval, label=y_eval), verbose_eval=100,
#               feval=gini_lgb, evals_result=eval_res, early_stopping_rounds=100)
# lgb_model.eval_valid(feval=gini_lgb)

gc.collect()
sub.head(2)
