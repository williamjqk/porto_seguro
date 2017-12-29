# %% this idea is from https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'

model_name = 'porto_seguro_dae019b'
data_path = '/home/ljc/data/porto_seguro_dae'
model_path = os.path.join(data_path, model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)

stored_model_name = 'porto_seguro_dae018b'
stored_model_meta = os.path.join(data_path, stored_model_name, 'dae_model.meta')
stored_model_path = os.path.join(data_path, stored_model_name)

# %% 创建整个supervised_nn和dataset，载入saver model并重新连接新的dataset和supervised_nn
import tensorflow as tf


nn_params = {
    "layers": [1000, 1000],
    "learning_rate": 1e-4, #1e-3, # 3e-3,
    "minibatch_size": 20000,#128,
    "learning_rate_decay": 0.995,
    "keep_rate": 0.6,
    "input_swap_noise": 0.05,
    "noise_std": 0.0,
    "n_epochs": 1#150 # 1000
}

in_dim = 221 # dimension of tfrecord features
BATCH_SIZE = nn_params['minibatch_size']
N_EPOCHS = nn_params['n_epochs']
learning_rate_decay = nn_params['learning_rate_decay']

def _parse_function(record):
    keys_to_features = {
        "rankgauss_feature": tf.FixedLenFeature((221,), tf.float32, default_value=tf.zeros((221,), dtype=tf.float32)),
        # "label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)
    return parsed_features["rankgauss_feature"]
    # return parsed_features["rankgauss_feature"], parsed_features["label"]

filenames = [
    # os.path.join(data_path, 'train_rankgauss.tfrecord'),
    os.path.join(data_path, 'test_rankgauss.tfrecord')
]

with tf.variable_scope("predict_model"):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    # dataset = dataset.shuffle(buffer_size=20000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat(N_EPOCHS)
    iterator = dataset.make_one_shot_iterator()
    next_feature = iterator.get_next()
    next_label = tf.placeholder(tf.int64, None)
    # next_feature, next_label = iterator.get_next()
    # <tf.Tensor 'supervised_nn_dataset/IteratorGetNext:0' shape=(?, 221) dtype=float32>
    # <tf.Tensor 'supervised_nn_dataset/IteratorGetNext:1' shape=(?,) dtype=int64>


sess = tf.Session()
saver = tf.train.import_meta_graph(stored_model_meta, input_map={"supervised_nn_dataset/IteratorGetNext:0": next_feature,
                                                                 "supervised_nn_dataset/IteratorGetNext:1": next_label, })
saver.restore(sess,tf.train.latest_checkpoint(stored_model_path))
graph = tf.get_default_graph()
# dae_h1 = graph.get_tensor_by_name("denoise_autoencoder/add_1:0")
# dae_h2 = graph.get_tensor_by_name("denoise_autoencoder/add_2:0")
# dae_h3 = graph.get_tensor_by_name("denoise_autoencoder/add_3:0")
traing_phase = graph.get_tensor_by_name('PlaceholderWithDefault_3:0')
bn_phase = graph.get_tensor_by_name('PlaceholderWithDefault_7:0')
y_pred_v = graph.get_tensor_by_name('Softmax:0')
# concat_x = tf.concat([dae_h1, dae_h2, dae_h3], axis=1)


y_out = np.zeros(892816)
# y_temp = sess.run(y_pred_v, feed_dict={traing_phase: False, next_label: np.random.rand(BATCH_SIZE)})
# y_out = np.hstack((y_out,y_temp[:,1]))
steps = 0
while True:
    try:
        y_temp = sess.run(y_pred_v, feed_dict={traing_phase: False, next_label: np.random.rand(BATCH_SIZE)})
        y_out[steps*BATCH_SIZE:steps*BATCH_SIZE+len(y_temp)] += y_temp[:,1]
        steps += 1
        if steps % 10 == 0:
            print(f"step: {steps}")
    except tf.errors.OutOfRangeError:
        break

# y_out
# array([ 0.046033  ,  0.05283122,  0.05432673, ...,  0.04590981,
#         0.05629953,  0.05042569])
# y_temp.shape

# %% output to submit
import pandas as pd
test_dat_0 = pd.read_csv(base_path+'test.csv')
sub = test_dat_0['id'].to_frame()
sub['target'] = 0
sub['target'] = y_out
sub.to_csv(os.path.join(model_path, 'test_'+model_name+'.csv'), index=False, float_format='%.5f')
