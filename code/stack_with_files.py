import pandas as pd
import numpy as np
from scipy.stats import skew
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, Lasso
from math import sqrt
from datetime import datetime
from collections import Counter
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import gc

base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'

test_xgb_ahar = pd.read_csv(base_path + 'xgb_submit_aharless.csv')
test_dnn = pd.read_csv(base_path + 'test_tune_camnugent_dnn.csv')
test_up = pd.read_csv(base_path + 'test_submission.csv')
test_cat = pd.read_csv(base_path + 'test_catboost_submission.csv')
test_kin = pd.read_csv(base_path + 'test_uberKinetics.csv')
test_gp = pd.read_csv(base_path + 'test_gpari.csv')
test_lgb_sc = pd.read_csv(base_path + 'test_sub_lgb_scale_impute.csv')

train_xgb_ahar = pd.read_csv(base_path + 'xgb_valid.csv')
train_dnn = pd.read_csv(base_path + 'train_tune_camnugent_dnn.csv')
train_up = pd.read_csv(base_path + 'train_submission.csv')
train_cat = pd.read_csv(base_path + 'train_catboost_submission.csv')
train_kin = pd.read_csv(base_path + 'train_uberKinetics.csv')
train_gp = pd.read_csv(base_path + 'train_gpari.csv')
train_lgb_sc = pd.read_csv(base_path + 'train_sub_lgb_scale_impute.csv')

train_df = train_xgb_ahar
train_df = train_df.rename(columns={'target': 'new0'})
train_df['new1'] = train_dnn['target']
train_df['new2'] = train_up['target']
train_df['new3'] = train_cat['target']
train_df['new4'] = train_kin['target']
train_df['new5'] = train_gp['target']
train_df['new6'] = train_lgb_sc['target']

train_0 = pd.read_csv(base_path + 'train.csv')
train_df['target']  = train_0['target']

test_df = test_xgb_ahar
test_df = test_df.rename(columns={'target': 'new0'})
test_df['new1'] = test_dnn['target'].rename('new0')
test_df['new2'] = test_up['target']
test_df['new3'] = test_cat['target']
test_df['new4'] = test_kin['target']
test_df['new5'] = test_gp['target']
test_df['new6'] = test_lgb_sc['target']

y = train_df['target'].values
testid= test_df['id'].values
sub = test_df['id'].to_frame()
sub['target'] = 0


# sub = pd.DataFrame()
# sub['id'] = testid

train_df.drop(['id','target'],axis=1,inplace=True)
X = train_df.as_matrix()#X_0
test_df.drop(['id'],axis=1,inplace=True)
X_test = test_df.as_matrix()


# %%
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

# %%
nrounds=10**6  # need to change to 2000
kfold = 5  # need to change to 5
seed = datetime.now().second + datetime.now().minute # 7
# np.random.seed(seed)
skf = StratifiedKFold(n_splits=kfold, random_state=seed)

# params = {'metric': 'auc', 'learning_rate' : 0.01, 'max_depth':8, 'max_bin':10,  'objective': 'binary',
#           'bagging_fraction':0.9,'bagging_freq':5,  'min_data': 500}
from sklearn import linear_model

logreg = linear_model.LinearRegression()
logreg.fit(X, y)

sub['target'] = logreg.predict(X_test)
test_df
sub['target'] = sub['target'].rank() / (1 * sub.shape[0])
dir(logreg)
logreg.get_params()



# eval_score_kfold = 0
# for i, (train_index, test_index) in enumerate(skf.split(X, y)):
#     print(' lgb kfold: {}  of  {} : '.format(i+1, kfold))
#     X_train, X_eval = X[train_index], X[test_index]
#     y_train, y_eval = y[train_index], y[test_index]
#     eval_res = dict()
#     lgb_model = lgb.train(params, lgb.Dataset(X_train, label=y_train), nrounds,
#                   lgb.Dataset(X_eval, label=y_eval), verbose_eval=100,
#                   feval=gini_lgb, evals_result=eval_res, early_stopping_rounds=100)
#     eval_score_kfold += eval_res['valid_0']['gini'][lgb_model.best_iteration-1] / (kfold)
#     sub['target'] += lgb_model.predict(X_test, # test[features].values
#                         num_iteration=lgb_model.best_iteration) / (kfold)
#     # sub_train['target'] += lgb_model.predict(Xnew_0, # train[features].values
#     #                     num_iteration=lgb_model.best_iteration) / (kfold)
# print(f'AVERAGE eval score: {eval_score_kfold}')
sub.to_csv(base_path+'stack_with_files_submit_20171031b.csv', index=False, float_format='%.5f')

# %%
import pandas as pd
import numpy as np
from scipy.stats import skew
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, Lasso
from math import sqrt
from datetime import datetime
from collections import Counter
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import gc

base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'

test_xgb_ahar = pd.read_csv(base_path + 'xgb_submit_aharless.csv')
test_up = pd.read_csv(base_path + 'test_submission.csv')
test_cat = pd.read_csv(base_path + 'test_catboost_submission.csv')
test_lgb = pd.read_csv(base_path + 'test_sub_lgb.csv')
test_xgb = pd.read_csv(base_path + 'test_sub_xgb.csv')

train_xgb_ahar = pd.read_csv(base_path + 'xgb_valid.csv')
train_up = pd.read_csv(base_path + 'train_submission.csv')
train_cat = pd.read_csv(base_path + 'train_catboost_submission.csv')
train_lgb = pd.read_csv(base_path + 'train_sub_lgb.csv')
train_xgb = pd.read_csv(base_path + 'train_sub_xgb.csv')

train_df = train_xgb_ahar
train_df = train_df.rename(columns={'target': 'new0'})
train_df['new1'] = train_up['target']
train_df['new2'] = train_cat['target']
train_df['new3'] = train_lgb['target']
train_df['new4'] = train_xgb['target']


train_0 = pd.read_csv(base_path + 'train.csv')
train_df['target']  = train_0['target']

test_df = test_xgb_ahar
test_df = test_df.rename(columns={'target': 'new0'})
test_df['new1'] = test_up['target']
test_df['new2'] = test_cat['target']
test_df['new3'] = test_lgb['target']
test_df['new4'] = test_xgb['target']

y = train_df['target'].values
testid= test_df['id'].values
sub = test_df['id'].to_frame()
sub['target'] = 0


# sub = pd.DataFrame()
# sub['id'] = testid

train_df.drop(['id','target'],axis=1,inplace=True)
X = train_df.as_matrix()#X_0
test_df.drop(['id'],axis=1,inplace=True)
X_test = test_df.as_matrix()


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

nrounds=10**6  # need to change to 2000
kfold = 5  # need to change to 5
seed = datetime.now().second + datetime.now().minute # 7
# np.random.seed(seed)
skf = StratifiedKFold(n_splits=kfold, random_state=seed)

# params = {'metric': 'auc', 'learning_rate' : 0.01, 'max_depth':3, 'max_bin':10,  'objective': 'binary',
#           'bagging_fraction':0.9,'bagging_freq':5,  'min_data': 500}
# cv:0.2872, lb:0.284
params = {'metric': 'auc', 'learning_rate' : 0.01, 'max_depth':8, 'max_bin':10,  'objective': 'binary',
          'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':5,  'min_data': 500}
# cv:0.2845 , lb:

# xgb_params = {
#     'seed': 0,
#     'colsample_bytree': 0.8,
#     'silent': 1,
#     'subsample': 0.6,
#     'learning_rate': 0.01,
#     'objective': 'reg:linear',
#     'max_depth': 1,
#     'num_parallel_tree': 1,
#     'min_child_weight': 1,
#     'eval_metric': 'rmse',
# }

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
    sub['target'] += lgb_model.predict(X_test, # test[features].values
                        num_iteration=lgb_model.best_iteration) / (kfold)
    # sub_train['target'] += lgb_model.predict(Xnew_0, # train[features].values
    #                     num_iteration=lgb_model.best_iteration) / (kfold)
print(f'AVERAGE eval score: {eval_score_kfold}')

sub.to_csv(base_path+'stack_with_files_submit_20171101c.csv', index=False, float_format='%.5f')
