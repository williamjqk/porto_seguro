# %% tensoflow https://www.kaggle.com/camnugent/deep-neural-network-insurance-claims-0-268
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'
test_dat = pd.read_csv(base_path + 'test.csv')
train_dat = pd.read_csv(base_path + 'train.csv')
submission = pd.read_csv(base_path + 'sample_submission.csv')

train_y = train_dat['target']
train_x = train_dat.drop(['target', 'id'], axis = 1)
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

scaler = StandardScaler()
scaled_numerics = scaler.fit_transform(merged_dat[numeric_features])
scaled_num_df = pd.DataFrame(scaled_numerics, columns =numeric_features )


merged_dat = merged_dat.drop(numeric_features, axis=1)

merged_dat = np.concatenate((merged_dat.values,scaled_num_df), axis = 1)

train_x = merged_dat[:train_x.shape[0]]
test_dat = merged_dat[train_x.shape[0]:]


config = tf.contrib.learn.RunConfig(tf_random_seed=42)

feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(train_x)

dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[1024,512,256], n_classes=2,
                                         feature_columns=feature_cols, config=config,
                                         optimizer=tf.train.AdamOptimizer(), dropout=0.2)

dnn_clf.fit(train_x, train_y, batch_size=50, steps=40000)

dnn_clf.evaluate(train_x, train_y)


dnn_y_pred = dnn_clf.predict_proba(test_dat)
dnn_y_pred_train = dnn_clf.predict_proba(train_x)


dnn_out = list(dnn_y_pred)

dnn_output = submission
dnn_output['target'] = [x[1] for x in dnn_out]


dnn_output.to_csv(base_path + 'test_tune_camnugent_dnn.csv', index=False, float_format='%.4f')

train = pd.read_csv(base_path + 'train.csv')
sub = pd.DataFrame()
sub['id'] = train['id']
sub['target'] = [x[1] for x in list(dnn_y_pred_train)]
sub.to_csv(base_path + 'train_tune_camnugent_dnn.csv', index=False, float_format='%.4f')
