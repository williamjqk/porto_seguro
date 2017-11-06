# %% MY PART (STACKING)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
# test files

test_xgb = pd.read_csv(base_path + 'test_sub_xgb.csv')
# test_lgb = pd.read_csv(base_path + 'test_sub_lgb.csv')
test_dnn = pd.read_csv(base_path + 'test_dnn_predictions.csv')
test_up = pd.read_csv(base_path + 'test_submission.csv')
test_cat = pd.read_csv(base_path + 'test_catboost_submission.csv')
test_kin = pd.read_csv(base_path + 'test_uberKinetics.csv')
test_gp = pd.read_csv(base_path + 'test_gpari.csv')
test_lgb_sc = pd.read_csv(base_path + 'test_sub_lgb_scale_impute.csv')
# test_tf_lgb_sc = pd.read_csv(base_path + 'test_sub_tf_lgb_scale_impute.csv')

test=pd.read_csv(base_path + 'test.csv')


test = pd.concat([test,
                   test_xgb[['target']].rename(columns = {'target' : 'xgb'}),
                #    test_lgb[['target']].rename(columns = {'target' : 'lgb'}),
                   test_dnn[['target']].rename(columns = {'target' : 'dnn'}),
                   test_up[['target']].rename(columns = {'target' : 'up'}),
                   test_cat[['target']].rename(columns = {'target' : 'cat'}),
                   test_kin[['target']].rename(columns = {'target' : 'kin'}),
                   test_gp[['target']].rename(columns = {'target' : 'gp'}),
                   test_lgb_sc[['target']].rename(columns = {'target' : 'lgb_sc'}),
                #    test_tf_lgb_sc[['target']].rename(columns = {'target' : 'tf_lgb_sc'})
                  ], axis = 1)


# train_cols = ['xgb', 'lgb', 'dnn', 'up', 'cat', 'kin', 'gp', 'lgb_sc', 'tf_lgb_sc']
# ### preprocess
# for t in train_cols:
#     test[t + '_rank'] = test[t].rank()
# test['target'] = (test['xgb_rank'] + test['lgb_rank'] + test['dnn_rank'] + test['up_rank'] + \
#                  test['cat_rank'] + test['kin_rank'] + test['gp_rank'] + test['lgb_sc_rank'] + test['tf_lgb_sc_rank']) / (9 * test.shape[0])

# train_cols = ['xgb', 'lgb', 'dnn', 'up', 'cat', 'kin', 'gp', 'tf_lgb_sc']
train_cols = ['xgb', 'dnn', 'up', 'cat', 'kin', 'gp', 'lgb_sc']
### preprocess
for t in train_cols:
    test[t + '_rank'] = test[t].rank()
test['target'] = (test['xgb_rank'] + test['dnn_rank'] + test['up_rank'] + \
                 test['cat_rank'] + test['kin_rank'] + test['gp_rank'] + test['lgb_sc_rank']) / (7 * test.shape[0])


# The final submission
test[['id', 'target']].to_csv(base_path + 'rank_avg_20171030a.csv.gz', index = False, compression = 'gzip')


# %% MY PART (STACKING)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
# test files

test_xgb_ahar = pd.read_csv(base_path + 'xgb_submit_aharless.csv')
# test_xgb = pd.read_csv(base_path + 'test_sub_xgb.csv')
# test_lgb = pd.read_csv(base_path + 'test_sub_lgb.csv')
test_dnn = pd.read_csv(base_path + 'test_dnn_predictions.csv')
test_up = pd.read_csv(base_path + 'test_submission.csv')
test_cat = pd.read_csv(base_path + 'test_catboost_submission.csv')
test_kin = pd.read_csv(base_path + 'test_uberKinetics.csv')
test_gp = pd.read_csv(base_path + 'test_gpari.csv')
test_lgb_sc = pd.read_csv(base_path + 'test_sub_lgb_scale_impute.csv')
# test_tf_lgb_sc = pd.read_csv(base_path + 'test_sub_tf_lgb_scale_impute.csv')

test=pd.read_csv(base_path + 'test.csv')


test = pd.concat([test,
                    test_xgb_ahar[['target']].rename(columns = {'target' : 'xgb_ahar'}),
                #    test_xgb[['target']].rename(columns = {'target' : 'xgb'}),
                #    test_lgb[['target']].rename(columns = {'target' : 'lgb'}),
                   test_dnn[['target']].rename(columns = {'target' : 'dnn'}),
                   test_up[['target']].rename(columns = {'target' : 'up'}),
                   test_cat[['target']].rename(columns = {'target' : 'cat'}),
                   test_kin[['target']].rename(columns = {'target' : 'kin'}),
                   test_gp[['target']].rename(columns = {'target' : 'gp'}),
                   test_lgb_sc[['target']].rename(columns = {'target' : 'lgb_sc'}),
                #    test_tf_lgb_sc[['target']].rename(columns = {'target' : 'tf_lgb_sc'})
                  ], axis = 1)


train_cols = ['xgb_ahar', 'dnn', 'up', 'cat', 'kin', 'gp', 'lgb_sc']
### preprocess
for t in train_cols:
    test[t + '_rank'] = test[t].rank()
test['target'] = (test['xgb_ahar_rank'] + test['dnn_rank'] + test['up_rank'] + \
                 test['cat_rank'] + test['kin_rank'] + test['gp_rank'] + test['lgb_sc_rank']) / (7 * test.shape[0])


# The final submission
test[['id', 'target']].to_csv(base_path + 'rank_avg_20171030b.csv.gz', index = False, compression = 'gzip')

# %% add output_kueipo: lb-0.285
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
# test files

test_xgb_ahar = pd.read_csv(base_path + 'xgb_submit_aharless.csv')
test_xgb_kueipo = pd.read_csv(base_path + 'output_kueipo.csv')
# test_xgb = pd.read_csv(base_path + 'test_sub_xgb.csv')
# test_lgb = pd.read_csv(base_path + 'test_sub_lgb.csv')
test_dnn = pd.read_csv(base_path + 'test_dnn_predictions.csv')
test_up = pd.read_csv(base_path + 'test_submission.csv')
test_cat = pd.read_csv(base_path + 'test_catboost_submission.csv')
test_kin = pd.read_csv(base_path + 'test_uberKinetics.csv')
test_gp = pd.read_csv(base_path + 'test_gpari.csv')
test_lgb_sc = pd.read_csv(base_path + 'test_sub_lgb_scale_impute.csv')
# test_tf_lgb_sc = pd.read_csv(base_path + 'test_sub_tf_lgb_scale_impute.csv')

test=pd.read_csv(base_path + 'test.csv')


test = pd.concat([test,
                    test_xgb_ahar[['target']].rename(columns = {'target' : 'xgb_ahar'}),
                    test_xgb_ahar[['target']].rename(columns = {'target' : 'xgb_kueipo'}),
                #    test_xgb[['target']].rename(columns = {'target' : 'xgb'}),
                #    test_lgb[['target']].rename(columns = {'target' : 'lgb'}),
                   test_dnn[['target']].rename(columns = {'target' : 'dnn'}),
                   test_up[['target']].rename(columns = {'target' : 'up'}),
                   test_cat[['target']].rename(columns = {'target' : 'cat'}),
                   test_kin[['target']].rename(columns = {'target' : 'kin'}),
                   test_gp[['target']].rename(columns = {'target' : 'gp'}),
                   test_lgb_sc[['target']].rename(columns = {'target' : 'lgb_sc'}),
                #    test_tf_lgb_sc[['target']].rename(columns = {'target' : 'tf_lgb_sc'})
                  ], axis = 1)


train_cols = ['xgb_ahar', 'xgb_kueipo', 'dnn', 'up', 'cat', 'kin', 'gp', 'lgb_sc']
### preprocess
for t in train_cols:
    test[t + '_rank'] = test[t].rank()
test['target'] = (test['xgb_ahar_rank'] + test['xgb_kueipo_rank'] + test['dnn_rank'] + test['up_rank'] + \
                 test['cat_rank'] + test['kin_rank'] + test['gp_rank'] + test['lgb_sc_rank']) / (8 * test.shape[0])


# The final submission
test[['id', 'target']].to_csv(base_path + 'rank_avg_20171030c.csv.gz', index = False, compression = 'gzip')

# %% remove camnugent's dnn: lb-0.285, don't improve rank than 'add output_kueipo'
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
# test files

test_xgb_ahar = pd.read_csv(base_path + 'xgb_submit_aharless.csv')
test_xgb_kueipo = pd.read_csv(base_path + 'output_kueipo.csv')
# test_xgb = pd.read_csv(base_path + 'test_sub_xgb.csv')
# test_lgb = pd.read_csv(base_path + 'test_sub_lgb.csv')
# test_dnn = pd.read_csv(base_path + 'test_dnn_predictions.csv')
test_up = pd.read_csv(base_path + 'test_submission.csv')
test_cat = pd.read_csv(base_path + 'test_catboost_submission.csv')
test_kin = pd.read_csv(base_path + 'test_uberKinetics.csv')
test_gp = pd.read_csv(base_path + 'test_gpari.csv')
test_lgb_sc = pd.read_csv(base_path + 'test_sub_lgb_scale_impute.csv')
# test_tf_lgb_sc = pd.read_csv(base_path + 'test_sub_tf_lgb_scale_impute.csv')

test=pd.read_csv(base_path + 'test.csv')


test = pd.concat([test,
                    test_xgb_ahar[['target']].rename(columns = {'target' : 'xgb_ahar'}),
                    test_xgb_ahar[['target']].rename(columns = {'target' : 'xgb_kueipo'}),
                #    test_xgb[['target']].rename(columns = {'target' : 'xgb'}),
                #    test_lgb[['target']].rename(columns = {'target' : 'lgb'}),
                #    test_dnn[['target']].rename(columns = {'target' : 'dnn'}),
                   test_up[['target']].rename(columns = {'target' : 'up'}),
                   test_cat[['target']].rename(columns = {'target' : 'cat'}),
                   test_kin[['target']].rename(columns = {'target' : 'kin'}),
                   test_gp[['target']].rename(columns = {'target' : 'gp'}),
                   test_lgb_sc[['target']].rename(columns = {'target' : 'lgb_sc'}),
                #    test_tf_lgb_sc[['target']].rename(columns = {'target' : 'tf_lgb_sc'})
                  ], axis = 1)


train_cols = ['xgb_ahar', 'xgb_kueipo', 'up', 'cat', 'kin', 'gp', 'lgb_sc']
### preprocess
for t in train_cols:
    test[t + '_rank'] = test[t].rank()
test['target'] = (test['xgb_ahar_rank'] + test['xgb_kueipo_rank'] + test['up_rank'] + \
                 test['cat_rank'] + test['kin_rank'] + test['gp_rank'] + test['lgb_sc_rank']) / (7 * test.shape[0])


# The final submission
test[['id', 'target']].to_csv(base_path + 'rank_avg_20171031a.csv.gz', index = False, compression = 'gzip')


# %% add tune camnugent dnn: 名次和分数都没有任何变化
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
# test files

test_xgb_ahar = pd.read_csv(base_path + 'xgb_submit_aharless.csv')
test_xgb_kueipo = pd.read_csv(base_path + 'output_kueipo.csv')
# test_xgb = pd.read_csv(base_path + 'test_sub_xgb.csv')
# test_lgb = pd.read_csv(base_path + 'test_sub_lgb.csv')
test_dnn = pd.read_csv(base_path + 'test_tune_camnugent_dnn.csv')
test_up = pd.read_csv(base_path + 'test_submission.csv')
test_cat = pd.read_csv(base_path + 'test_catboost_submission.csv')
test_kin = pd.read_csv(base_path + 'test_uberKinetics.csv')
test_gp = pd.read_csv(base_path + 'test_gpari.csv')
test_lgb_sc = pd.read_csv(base_path + 'test_sub_lgb_scale_impute.csv')
# test_tf_lgb_sc = pd.read_csv(base_path + 'test_sub_tf_lgb_scale_impute.csv')

test=pd.read_csv(base_path + 'test.csv')


test = pd.concat([test,
                    test_xgb_ahar[['target']].rename(columns = {'target' : 'xgb_ahar'}),
                    test_xgb_ahar[['target']].rename(columns = {'target' : 'xgb_kueipo'}),
                #    test_xgb[['target']].rename(columns = {'target' : 'xgb'}),
                #    test_lgb[['target']].rename(columns = {'target' : 'lgb'}),
                   test_dnn[['target']].rename(columns = {'target' : 'dnn'}),
                   test_up[['target']].rename(columns = {'target' : 'up'}),
                   test_cat[['target']].rename(columns = {'target' : 'cat'}),
                   test_kin[['target']].rename(columns = {'target' : 'kin'}),
                   test_gp[['target']].rename(columns = {'target' : 'gp'}),
                   test_lgb_sc[['target']].rename(columns = {'target' : 'lgb_sc'}),
                #    test_tf_lgb_sc[['target']].rename(columns = {'target' : 'tf_lgb_sc'})
                  ], axis = 1)


train_cols = ['xgb_ahar', 'xgb_kueipo', 'dnn', 'up', 'cat', 'kin', 'gp', 'lgb_sc']
### preprocess
for t in train_cols:
    test[t + '_rank'] = test[t].rank()
test['target'] = (test['xgb_ahar_rank'] + test['xgb_kueipo_rank'] + test['dnn_rank'] + test['up_rank'] + \
                 test['cat_rank'] + test['kin_rank'] + test['gp_rank'] + test['lgb_sc_rank']) / (8 * test.shape[0])


# The final submission
test[['id', 'target']].to_csv(base_path + 'rank_avg_20171031b.csv.gz', index = False, compression = 'gzip')
