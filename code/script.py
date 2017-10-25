# %%
base_path = '../input/' # your folder

import pandas as pd
train=pd.read_csv(base_path + 'train.csv')
test=pd.read_csv(base_path + 'test.csv')

train['ps_ind_0609_bin'] = train.apply(lambda x: 1 if x['ps_ind_06_bin'] == 1 else (2 if x['ps_ind_07_bin'] == 1 else
(
3 if x['ps_ind_08_bin'] == 1 else (4 if x['ps_ind_09_bin'] == 1 else 5)

)), axis = 1)

test['ps_ind_0609_bin'] = test.apply(lambda x: 1 if x['ps_ind_06_bin'] == 1 else (2 if x['ps_ind_07_bin'] == 1 else
(
3 if x['ps_ind_08_bin'] == 1 else (4 if x['ps_ind_09_bin'] == 1 else 5)

)), axis = 1)

train.drop(['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin'], axis = 1, inplace = True)

test.drop(['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin'], axis = 1, inplace = True)

train['ps_car_13'] = (train['ps_car_13']*train['ps_car_13']* 48400).round(0)

test['ps_car_13'] = (test['ps_car_13']*test['ps_car_13']* 48400).round(0)

train['ps_car_12'] = (train['ps_car_12']*train['ps_car_12']).round(4) * 10000

test['ps_car_12'] = (test['ps_car_12']*test['ps_car_12']).round(4) * 10000

for c in train[[c for c in train.columns if 'bin' in c]].columns:
    for cc in train[[c for c in train.columns if 'bin' in c]].columns:
            if train[train[cc] * train[c] == 0].shape[0] == train.shape[0]:
                print(c, cc)

train['ps_ind_161718_bin'] = train.apply(lambda x: 1 if x['ps_ind_16_bin'] == 1 else
                                        (2 if x['ps_ind_17_bin'] == 1 else 3), axis = 1
                                        )

test['ps_ind_161718_bin'] = test.apply(lambda x: 1 if x['ps_ind_16_bin'] == 1 else
                                        (2 if x['ps_ind_17_bin'] == 1 else 3), axis = 1
                                        )

train.drop(['ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin'], axis = 1, inplace = True)

test.drop(['ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin'], axis = 1, inplace = True)

train.to_csv(base_path + 'train_p.csv', index = False)

test.to_csv(base_path + 'test_p.csv', index = False)
