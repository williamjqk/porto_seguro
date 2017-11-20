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

# %% add output_kueipo: lb-0.285 # my best lb (2017-11-07)
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

# %% remove camnugent's dnn: lb-0.285, don't improve rank than 'add output_kueipo' (说明没有best的好)
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


# %% add tune camnugent dnn: 名次和分数都没有任何变化(说明没有best的好)
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

# %% replace kueipo and dnn: lb-0.285 # my best lb (325->293) (2017-11-07)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
# test files

test_xgb_ahar = pd.read_csv(base_path + 'xgb_submit_aharless.csv')
test_xgb_kueipo = pd.read_csv(base_path + 'test_kueipo_v2.csv')
# test_xgb = pd.read_csv(base_path + 'test_sub_xgb.csv')
# test_lgb = pd.read_csv(base_path + 'test_sub_lgb.csv')
test_dnn = pd.read_csv(base_path + 'test_dnn_camnugent_fe_v2004.csv')
test_up = pd.read_csv(base_path + 'test_submission.csv') # script_v2 里面ogrellier生成的代码
test_cat = pd.read_csv(base_path + 'test_catboost_submission.csv')
test_kin = pd.read_csv(base_path + 'test_uberKinetics.csv')
test_gp = pd.read_csv(base_path + 'test_gpari.csv')
test_lgb_sc = pd.read_csv(base_path + 'test_sub_lgb_scale_impute.csv') # code/script_tf_lgb_v1001.py生成的代码
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
test[['id', 'target']].to_csv(base_path + 'rank_avg_20171107a.csv.gz', index = False, compression = 'gzip')


# %% add median_rank_submission.csv: 名次没动 肯定不如rank_avg_20171107a 好
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
# test files

test_xgb_ahar = pd.read_csv(base_path + 'xgb_submit_aharless.csv')
test_xgb_kueipo = pd.read_csv(base_path + 'test_kueipo_v2.csv')
# test_xgb = pd.read_csv(base_path + 'test_sub_xgb.csv')
# test_lgb = pd.read_csv(base_path + 'test_sub_lgb.csv')
test_dnn = pd.read_csv(base_path + 'test_dnn_camnugent_fe_v2004.csv')
test_up = pd.read_csv(base_path + 'test_submission.csv')
test_cat = pd.read_csv(base_path + 'test_catboost_submission.csv')
test_kin = pd.read_csv(base_path + 'test_uberKinetics.csv')
test_gp = pd.read_csv(base_path + 'test_gpari.csv')
test_lgb_sc = pd.read_csv(base_path + 'test_sub_lgb_scale_impute.csv')
# test_tf_lgb_sc = pd.read_csv(base_path + 'test_sub_tf_lgb_scale_impute.csv')
test_kernels20 = pd.read_csv(base_path + 'median_rank_submission.csv')

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
                   test_kernels20[['target']].rename(columns = {'target' : 'kernels20'}),
                #    test_tf_lgb_sc[['target']].rename(columns = {'target' : 'tf_lgb_sc'})
                  ], axis = 1)


train_cols = ['xgb_ahar', 'xgb_kueipo', 'dnn', 'up', 'cat', 'kin', 'gp', 'lgb_sc','kernels20']
### preprocess
for t in train_cols:
    test[t + '_rank'] = test[t].rank()
test['target'] = (test['xgb_ahar_rank'] + test['xgb_kueipo_rank'] + test['dnn_rank'] + test['up_rank'] + \
                 test['cat_rank'] + test['kin_rank'] + test['gp_rank'] + test['lgb_sc_rank'] + 20*test['kernels20_rank']) / (28 * test.shape[0])
# The final submission
test[['id', 'target']].to_csv(base_path + 'rank_avg_20171107b.csv.gz', index = False, compression = 'gzip')

# %% 多个.csv取中值median试试： 0.285， 相比rank_avg_20171107a没有任何提高
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
# test files

test_xgb_ahar = pd.read_csv(base_path + 'xgb_submit_aharless.csv')
test_xgb_kueipo = pd.read_csv(base_path + 'test_kueipo_v2.csv')
# test_xgb = pd.read_csv(base_path + 'test_sub_xgb.csv')
# test_lgb = pd.read_csv(base_path + 'test_sub_lgb.csv')
test_dnn = pd.read_csv(base_path + 'test_dnn_camnugent_fe_v2004.csv')
test_up = pd.read_csv(base_path + 'test_submission.csv')
test_cat = pd.read_csv(base_path + 'test_catboost_submission.csv')
test_kin = pd.read_csv(base_path + 'test_uberKinetics.csv')
test_gp = pd.read_csv(base_path + 'test_gpari.csv')
test_lgb_sc = pd.read_csv(base_path + 'test_sub_lgb_scale_impute.csv')
# test_tf_lgb_sc = pd.read_csv(base_path + 'test_sub_tf_lgb_scale_impute.csv')
test_kernels20 = pd.read_csv(base_path + 'median_rank_submission.csv')

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
                   test_kernels20[['target']].rename(columns = {'target' : 'kernels20'}),
                #    test_tf_lgb_sc[['target']].rename(columns = {'target' : 'tf_lgb_sc'})
                  ], axis = 1)


train_cols = ['xgb_ahar', 'xgb_kueipo', 'dnn', 'up', 'cat', 'kin', 'gp', 'lgb_sc','kernels20']
### preprocess
for t in train_cols:
    test[t + '_rank'] = test[t].rank()
test['target'] = pd.concat([test['xgb_ahar_rank'], test['xgb_kueipo_rank'], test['dnn_rank'], test['up_rank'], \
                 test['cat_rank'], test['kin_rank'], test['gp_rank'], test['lgb_sc_rank'], test['kernels20_rank']], axis=1).median(axis=1) / (test.shape[0])
# The final submission
test[['id', 'target']].to_csv(base_path + 'rank_avg_20171107c.csv.gz', index = False, compression = 'gzip')

# %% 跟rank_avg_20171107a几乎一样，只是用的median： 0.285，相比rank_avg_20171107a名次没有任何提高
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
# test files

test_xgb_ahar = pd.read_csv(base_path + 'xgb_submit_aharless.csv')
test_xgb_kueipo = pd.read_csv(base_path + 'test_kueipo_v2.csv')
# test_xgb = pd.read_csv(base_path + 'test_sub_xgb.csv')
# test_lgb = pd.read_csv(base_path + 'test_sub_lgb.csv')
test_dnn = pd.read_csv(base_path + 'test_dnn_camnugent_fe_v2004.csv')
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
test['target'] = pd.concat([test['xgb_ahar_rank'], test['xgb_kueipo_rank'], test['dnn_rank'], test['up_rank'], \
                 test['cat_rank'], test['kin_rank'], test['gp_rank'], test['lgb_sc_rank']], axis=1).median(axis=1) / (test.shape[0])
# The final submission
test[['id', 'target']].to_csv(base_path + 'rank_avg_20171107d.csv.gz', index = False, compression = 'gzip')

# %% lb:0.284 （比rank_avg_20171107a.csv.gz差好多）
# replace 'test_dnn_camnugent_fe_v2004' with 'test_aquatic_NNShallow_5fold_3runs_sub.csv'
# replace 'test_submission.csv' with 'test_ogrellier_xgb.csv'
# add test_sub_xgb.csv
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
# test files

test_xgb_ahar = pd.read_csv(base_path + 'xgb_submit_aharless.csv')
test_xgb_kueipo = pd.read_csv(base_path + 'test_kueipo_v2.csv')
test_xgb = pd.read_csv(base_path + 'test_sub_xgb.csv')
# test_lgb = pd.read_csv(base_path + 'test_sub_lgb.csv')
# test_dnn = pd.read_csv(base_path + 'test_dnn_camnugent_fe_v2004.csv')
# test_up = pd.read_csv(base_path + 'test_submission.csv') # script_v2 里面ogrellier生成的代码
test_cat = pd.read_csv(base_path + 'test_catboost_submission.csv')
test_kin = pd.read_csv(base_path + 'test_uberKinetics.csv')
test_gp = pd.read_csv(base_path + 'test_gpari.csv')
test_lgb_sc = pd.read_csv(base_path + 'test_sub_lgb_scale_impute.csv') # code/script_tf_lgb_v1001.py生成的代码
# test_tf_lgb_sc = pd.read_csv(base_path + 'test_sub_tf_lgb_scale_impute.csv')
test_ogrellier_xgb = pd.read_csv(base_path + 'test_ogrellier_xgb.csv')
test_aquatic_nn = pd.read_csv(base_path + 'test_aquatic_NNShallow_5fold_3runs_sub.csv')

test=pd.read_csv(base_path + 'test.csv')


test = pd.concat([test,
                    test_xgb_ahar[['target']].rename(columns = {'target' : 'xgb_ahar'}),
                    test_xgb_kueipo[['target']].rename(columns = {'target' : 'xgb_kueipo'}),
                   test_xgb[['target']].rename(columns = {'target' : 'xgb'}),
                #    test_lgb[['target']].rename(columns = {'target' : 'lgb'}),
                #    test_dnn[['target']].rename(columns = {'target' : 'dnn'}),
                #    test_up[['target']].rename(columns = {'target' : 'up'}),
                   test_cat[['target']].rename(columns = {'target' : 'cat'}),
                   test_kin[['target']].rename(columns = {'target' : 'kin'}),
                   test_gp[['target']].rename(columns = {'target' : 'gp'}),
                   test_lgb_sc[['target']].rename(columns = {'target' : 'lgb_sc'}),
                #    test_tf_lgb_sc[['target']].rename(columns = {'target' : 'tf_lgb_sc'})
                   test_ogrellier_xgb[['target']].rename(columns = {'target' : 'ogrellier_xgb'}),
                   test_aquatic_nn[['target']].rename(columns = {'target' : 'aquatic_nn'}),
                  ], axis = 1)


train_cols = ['xgb_ahar', 'xgb_kueipo', 'xgb', 'cat', 'kin', 'gp', 'lgb_sc', 'ogrellier_xgb', 'aquatic_nn']
### preprocess
for t in train_cols:
    test[t + '_rank'] = test[t].rank()
test['target'] = (test['xgb_ahar_rank'] + test['xgb_kueipo_rank'] + test['xgb_rank'] + \
                 test['cat_rank'] + test['kin_rank'] + test['gp_rank'] + test['lgb_sc_rank'] +\
                 test['ogrellier_xgb_rank'] + test['aquatic_nn_rank']) / (len(train_cols) * test.shape[0])
# The final submission
test[['id', 'target']].to_csv(base_path + 'rank_avg_20171108a.csv.gz', index = False, compression = 'gzip')


# %% 修改了rank_avg_20171107a.csv.gz两个xgb_ahar的bug： lb:0.285(名次没有提升)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
# test files

test_xgb_ahar = pd.read_csv(base_path + 'xgb_submit_aharless.csv')
test_xgb_kueipo = pd.read_csv(base_path + 'test_kueipo_v2.csv')
# test_xgb = pd.read_csv(base_path + 'test_sub_xgb.csv')
# test_lgb = pd.read_csv(base_path + 'test_sub_lgb.csv')
test_dnn = pd.read_csv(base_path + 'test_dnn_camnugent_fe_v2004.csv')
test_up = pd.read_csv(base_path + 'test_submission.csv') # script_v2 里面ogrellier生成的代码
test_cat = pd.read_csv(base_path + 'test_catboost_submission.csv')
test_kin = pd.read_csv(base_path + 'test_uberKinetics.csv')
test_gp = pd.read_csv(base_path + 'test_gpari.csv')
test_lgb_sc = pd.read_csv(base_path + 'test_sub_lgb_scale_impute.csv') # code/script_tf_lgb_v1001.py生成的代码
# test_tf_lgb_sc = pd.read_csv(base_path + 'test_sub_tf_lgb_scale_impute.csv')

test=pd.read_csv(base_path + 'test.csv')


test = pd.concat([test,
                    test_xgb_ahar[['target']].rename(columns = {'target' : 'xgb_ahar'}),
                    test_xgb_kueipo[['target']].rename(columns = {'target' : 'xgb_kueipo'}),
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
test[['id', 'target']].to_csv(base_path + 'rank_avg_20171108b.csv.gz', index = False, compression = 'gzip')

# %% 相比rank_avg_20171107a.csv.gz加强了xgb_ahar, lb:0.285, lb-rank:302->287, mybest:20171108
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
# test files

test_xgb_ahar = pd.read_csv(base_path + 'xgb_submit_aharless.csv')
test_xgb_kueipo = pd.read_csv(base_path + 'test_kueipo_v2.csv')
# test_xgb = pd.read_csv(base_path + 'test_sub_xgb.csv')
# test_lgb = pd.read_csv(base_path + 'test_sub_lgb.csv')
test_dnn = pd.read_csv(base_path + 'test_dnn_camnugent_fe_v2004.csv')
test_up = pd.read_csv(base_path + 'test_submission.csv') # script_v2 里面ogrellier生成的代码
test_cat = pd.read_csv(base_path + 'test_catboost_submission.csv')
test_kin = pd.read_csv(base_path + 'test_uberKinetics.csv')
test_gp = pd.read_csv(base_path + 'test_gpari.csv')
test_lgb_sc = pd.read_csv(base_path + 'test_sub_lgb_scale_impute.csv') # code/script_tf_lgb_v1001.py生成的代码
# test_tf_lgb_sc = pd.read_csv(base_path + 'test_sub_tf_lgb_scale_impute.csv')

test=pd.read_csv(base_path + 'test.csv')


test = pd.concat([test,
                    test_xgb_ahar[['target']].rename(columns = {'target' : 'xgb_ahar'}),
                    test_xgb_kueipo[['target']].rename(columns = {'target' : 'xgb_kueipo'}),
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
test['target'] = (2.5*test['xgb_ahar_rank'] + test['xgb_kueipo_rank'] + test['dnn_rank'] + test['up_rank'] + \
                 test['cat_rank'] + test['kin_rank'] + test['gp_rank'] + test['lgb_sc_rank']) / (9.5 * test.shape[0])
# The final submission
test[['id', 'target']].to_csv(base_path + 'rank_avg_20171108c.csv.gz', index = False, compression = 'gzip')

# %% 相比rank_avg_20171108c.csv.gz替换了v2004->v2006, 名次没有任何提升
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
# test files

test_xgb_ahar = pd.read_csv(base_path + 'xgb_submit_aharless.csv')
test_xgb_kueipo = pd.read_csv(base_path + 'test_kueipo_v2.csv')
# test_xgb = pd.read_csv(base_path + 'test_sub_xgb.csv')
# test_lgb = pd.read_csv(base_path + 'test_sub_lgb.csv')
test_dnn = pd.read_csv(base_path + 'test_dnn_camnugent_fe_v2006.csv')
test_up = pd.read_csv(base_path + 'test_submission.csv') # script_v2 里面ogrellier生成的代码
test_cat = pd.read_csv(base_path + 'test_catboost_submission.csv')
test_kin = pd.read_csv(base_path + 'test_uberKinetics.csv')
test_gp = pd.read_csv(base_path + 'test_gpari.csv')
test_lgb_sc = pd.read_csv(base_path + 'test_sub_lgb_scale_impute.csv') # code/script_tf_lgb_v1001.py生成的代码
# test_tf_lgb_sc = pd.read_csv(base_path + 'test_sub_tf_lgb_scale_impute.csv')

test=pd.read_csv(base_path + 'test.csv')

test = pd.concat([test,
                    test_xgb_ahar[['target']].rename(columns = {'target' : 'xgb_ahar'}),
                    test_xgb_kueipo[['target']].rename(columns = {'target' : 'xgb_kueipo'}),
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
test['target'] = (2.5*test['xgb_ahar_rank'] + test['xgb_kueipo_rank'] + test['dnn_rank'] + test['up_rank'] + \
                 test['cat_rank'] + test['kin_rank'] + test['gp_rank'] + test['lgb_sc_rank']) / (9.5 * test.shape[0])
# The final submission
test[['id', 'target']].to_csv(base_path + 'rank_avg_20171111a.csv.gz', index = False, compression = 'gzip')

# %% 只有lb:0.286
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
import pandas as pd
import numpy as np


vpaslay_kagglemix = pd.read_csv(base_path + 'vpaslay_kagglemix.csv')
tunguz_sub = pd.read_csv(base_path + 'tunguz_sub.csv.gz', compression = 'gzip')
serigne_sub_harmonic = pd.read_csv(base_path + 'serigne_sub_harmonic.csv')
# median_rank_submission = pd.read_csv('../input/median-rank-submission/median_rank_submission.csv')

# Ensemble and create submission

sub = pd.DataFrame()
sub['id'] = vpaslay_kagglemix['id']
sub['target'] = np.exp(np.mean(
	[
	vpaslay_kagglemix['target'].apply(lambda x: np.log(x)),\
	tunguz_sub['target'].apply(lambda x: np.log(x)),\
	serigne_sub_harmonic['target'].apply(lambda x: np.log(x)),\
	# median_rank_submission['target'].apply(lambda x: np.log(x))\
	], axis =0))

sub.to_csv(base_path + 'rank_avg_20171115a.csv.gz', index = False, compression = 'gzip')

# %% rank_avg_20171108c.csv.gz增加了kagglemix.csv, lb:0.285(kagglemix的权重6或20都是这个分数)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
# test files

test_xgb_ahar = pd.read_csv(base_path + 'xgb_submit_aharless.csv')
test_xgb_kueipo = pd.read_csv(base_path + 'test_kueipo_v2.csv')
# test_xgb = pd.read_csv(base_path + 'test_sub_xgb.csv')
# test_lgb = pd.read_csv(base_path + 'test_sub_lgb.csv')
test_dnn = pd.read_csv(base_path + 'test_dnn_camnugent_fe_v2004.csv')
test_up = pd.read_csv(base_path + 'test_submission.csv') # script_v2 里面ogrellier生成的代码
test_cat = pd.read_csv(base_path + 'test_catboost_submission.csv')
test_kin = pd.read_csv(base_path + 'test_uberKinetics.csv')
test_gp = pd.read_csv(base_path + 'test_gpari.csv')
test_lgb_sc = pd.read_csv(base_path + 'test_sub_lgb_scale_impute.csv') # code/script_tf_lgb_v1001.py生成的代码
# test_tf_lgb_sc = pd.read_csv(base_path + 'test_sub_tf_lgb_scale_impute.csv')
test_vpaslay_mix = pd.read_csv(base_path + 'vpaslay_kagglemix.csv')

test=pd.read_csv(base_path + 'test.csv')


test = pd.concat([test,
                    test_xgb_ahar[['target']].rename(columns = {'target' : 'xgb_ahar'}),
                    test_xgb_kueipo[['target']].rename(columns = {'target' : 'xgb_kueipo'}),
                #    test_xgb[['target']].rename(columns = {'target' : 'xgb'}),
                #    test_lgb[['target']].rename(columns = {'target' : 'lgb'}),
                   test_dnn[['target']].rename(columns = {'target' : 'dnn'}),
                   test_up[['target']].rename(columns = {'target' : 'up'}),
                   test_cat[['target']].rename(columns = {'target' : 'cat'}),
                   test_kin[['target']].rename(columns = {'target' : 'kin'}),
                   test_gp[['target']].rename(columns = {'target' : 'gp'}),
                   test_lgb_sc[['target']].rename(columns = {'target' : 'lgb_sc'}),
                #    test_tf_lgb_sc[['target']].rename(columns = {'target' : 'tf_lgb_sc'})
                    test_vpaslay_mix[['target']].rename(columns = {'target' : 'vpaslay_mix'}),
                  ], axis = 1)


train_cols = ['xgb_ahar', 'xgb_kueipo', 'dnn', 'up', 'cat', 'kin', 'gp', 'lgb_sc', 'vpaslay_mix']
### preprocess
for t in train_cols:
    test[t + '_rank'] = test[t].rank()
test['target'] = (2.5*test['xgb_ahar_rank'] + test['xgb_kueipo_rank'] + test['dnn_rank'] + test['up_rank'] + \
                 test['cat_rank'] + test['kin_rank'] + test['gp_rank'] + test['lgb_sc_rank'] +\
                 20*test['vpaslay_mix'] ) / ((2.5+7+20) * test.shape[0])
# The final submission
test[['id', 'target']].to_csv(base_path + 'rank_avg_20171115b.csv.gz', index = False, compression = 'gzip')
