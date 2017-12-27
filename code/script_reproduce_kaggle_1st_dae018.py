# %% this idea is from https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'

model_name = 'porto_seguro_dae018'
data_path = '/home/ljc/data/porto_seguro_dae'
model_path = os.path.join(data_path, model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)

stored_model_name = 'porto_seguro_dae017'
stored_model_meta = os.path.join(data_path, stored_model_name, 'dae_model.meta')
stored_model_path = os.path.join(data_path, stored_model_name)

# # %%
# import tensorflow as tf
#
# sess = tf.Session()
# saver = tf.train.import_meta_graph(stored_model_meta)
# saver.restore(sess,tf.train.latest_checkpoint(stored_model_path))
#
# graph = tf.get_default_graph()
# w1 = graph.get_tensor_by_name("denoise_autoencoder/layer1_dense/kernel:0")
# dae_x = graph.get_tensor_by_name("denoise_autoencoder/add:0")
# dae_h1 = graph.get_tensor_by_name("denoise_autoencoder/add_1:0")
# dae_h2 = graph.get_tensor_by_name("denoise_autoencoder/add_2:0")
# dae_h3 = graph.get_tensor_by_name("denoise_autoencoder/add_3:0")
# traing_phase = graph.get_tensor_by_name('PlaceholderWithDefault_3:0')
#
#
# import numpy as np
# x = np.random.rand(128,221)
# x1,x2,x3 = sess.run([dae_h1, dae_h2, dae_h3], feed_dict={dae_x: x, traing_phase: False})
# print(x1)
#
# # x1,x2,x3 = sess.run([dae_h1, dae_h2, dae_h3], feed_dict={dae_x: x, traing_phase: False})
# # print(x1.max(),x1.min(),x2.max(),x2.min(),x3.max(),x3.min())
# # x1 = sess.run([dae_h1], feed_dict={dae_x: x, traing_phase: False})
# # print(x1)
# # sess.run([w1], feed_dict={dae_x: x})


# # %% from tfrecord to restored model
# dae_params = {
#     "layers": [1500, 1500, 1500],
#     "learning_rate": 3e-3, # 3e-3,
#     "minibatch_size": 128,
#     "learning_rate_decay": 0.995,
#     "keep_rate": 0.8,
#     "input_swap_noise": 0.15,
#     "noise_std": 0.0,
#     "n_epochs": 1000 # 1000
# }
# in_dim = 221 # dimension of tfrecord features
# BATCH_SIZE = dae_params['minibatch_size']
# N_EPOCHS = dae_params['n_epochs']
# learning_rate_decay = dae_params['learning_rate_decay']
#
# import tensorflow as tf
# # tf.reset_default_graph()
# def _parse_function(record):
#     keys_to_features = {
#         "rankgauss_feature": tf.FixedLenFeature((221,), tf.float32, default_value=tf.zeros((221,), dtype=tf.float32)),
#         "label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
#     }
#     parsed_features = tf.parse_single_example(record, keys_to_features)
#     return parsed_features["rankgauss_feature"], parsed_features["label"]
#
# filenames = [
#     os.path.join(data_path, 'train_rankgauss.tfrecord'),
#     # os.path.join(data_path, 'test_rankgauss.tfrecord')
# ]
#
# dataset = tf.data.TFRecordDataset(filenames)
# dataset = dataset.map(_parse_function)
# dataset = dataset.shuffle(buffer_size=200000)
# dataset = dataset.batch(BATCH_SIZE)
# dataset = dataset.repeat(N_EPOCHS)
# iterator = dataset.make_one_shot_iterator()
# next_feature, next_label = iterator.get_next()
#
# tf_x1 = tf.placeholder(tf.float32, [None, 221])
#
# inp2_name = "denoise_autoencoder/add:0"
# out3_name_l = ["denoise_autoencoder/add_1:0", "denoise_autoencoder/add_2:0", "denoise_autoencoder/add_3:0"]
# with graph.as_default():
#     dae_h1, dae_h2, dae_h3 = tf.import_graph_def(graph.as_graph_def(), input_map={inp2_name: next_feature}, return_elements=out3_name_l)
# sess.run([dae_h1, dae_h2, dae_h3]) # cannot work, because there is a 'import/' prefix before node names

# # %% 载入saver model并重新连接tensor，使用placeholder作为用例
# import tensorflow as tf
# tf.reset_default_graph()
# sess = tf.Session()
#
# tf_x1 = tf.placeholder(tf.float32, [None, 221])
# # saver = tf.train.import_meta_graph(stored_model_meta, input_map={"denoise_autoencoder/add:0": next_feature})
# saver = tf.train.import_meta_graph(stored_model_meta, input_map={"denoise_autoencoder/add:0": tf_x1})
# saver.restore(sess,tf.train.latest_checkpoint(stored_model_path))
#
# graph = tf.get_default_graph()
# # dae_x = graph.get_tensor_by_name("denoise_autoencoder/add:0")
# dae_h1 = graph.get_tensor_by_name("denoise_autoencoder/add_1:0")
# dae_h2 = graph.get_tensor_by_name("denoise_autoencoder/add_2:0")
# dae_h3 = graph.get_tensor_by_name("denoise_autoencoder/add_3:0")
# traing_phase = graph.get_tensor_by_name('PlaceholderWithDefault_3:0')
# # x1,x2,x3 = sess.run([dae_h1, dae_h2, dae_h3], feed_dict={traing_phase: False})
# x1,x2,x3 = sess.run([dae_h1, dae_h2, dae_h3], feed_dict={tf_x1: np.zeros([20,221]), traing_phase: False})
# print(x1);print(x2);print(x3)

# # %% 载入saver model并重新连接tensor，使用TFRecordDatasetzu作为用例
# dae_params = {
#     "layers": [1500, 1500, 1500],
#     "learning_rate": 3e-3, # 3e-3,
#     "minibatch_size": 128,
#     "learning_rate_decay": 0.995,
#     "keep_rate": 0.8,
#     "input_swap_noise": 0.15,
#     "noise_std": 0.0,
#     "n_epochs": 1000 # 1000
# }
# in_dim = 221 # dimension of tfrecord features
# BATCH_SIZE = dae_params['minibatch_size']
# N_EPOCHS = dae_params['n_epochs']
# learning_rate_decay = dae_params['learning_rate_decay']
#
# import tensorflow as tf
# # tf.reset_default_graph()
# def _parse_function(record):
#     keys_to_features = {
#         "rankgauss_feature": tf.FixedLenFeature((221,), tf.float32, default_value=tf.zeros((221,), dtype=tf.float32)),
#         "label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
#     }
#     parsed_features = tf.parse_single_example(record, keys_to_features)
#     # return parsed_features["rankgauss_feature"]
#     return parsed_features["rankgauss_feature"], parsed_features["label"]
#
# filenames = [
#     os.path.join(data_path, 'train_rankgauss.tfrecord'),
#     # os.path.join(data_path, 'test_rankgauss.tfrecord')
# ]
#
# with tf.variable_scope("supervised_nn_dataset"):
#     dataset = tf.data.TFRecordDataset(filenames)
#     dataset = dataset.map(_parse_function)
#     dataset = dataset.shuffle(buffer_size=200000)
#     dataset = dataset.batch(BATCH_SIZE)
#     dataset = dataset.repeat(N_EPOCHS)
#     iterator = dataset.make_one_shot_iterator()
#     next_feature, next_label = iterator.get_next()
#
# sess = tf.Session()
# saver = tf.train.import_meta_graph(stored_model_meta, input_map={"denoise_autoencoder/add:0": next_feature})
# saver.restore(sess,tf.train.latest_checkpoint(stored_model_path))
#
# graph = tf.get_default_graph()
# # dae_x = graph.get_tensor_by_name("denoise_autoencoder/add:0")
# dae_h1 = graph.get_tensor_by_name("denoise_autoencoder/add_1:0")
# dae_h2 = graph.get_tensor_by_name("denoise_autoencoder/add_2:0")
# dae_h3 = graph.get_tensor_by_name("denoise_autoencoder/add_3:0")
# traing_phase = graph.get_tensor_by_name('PlaceholderWithDefault_3:0')
# x1,x2,x3 = sess.run([dae_h1, dae_h2, dae_h3], feed_dict={traing_phase: False})
# print(x1);print(x2);print(x3)




# %% 创建整个supervised_nn和dataset，载入saver model并重新连接新的dataset和supervised_nn
import tensorflow as tf

# dae_params = {
#     "layers": [1500, 1500, 1500],
#     "learning_rate": 3e-3, # 3e-3,
#     "minibatch_size": 128,
#     "learning_rate_decay": 0.995,
#     "keep_rate": 0.8,
#     "input_swap_noise": 0.15,
#     "noise_std": 0.0,
#     "n_epochs": 1000 # 1000
# }

nn_params = {
    "layers": [1000, 1000],
    "learning_rate": 1e-4, #1e-3, # 3e-3,
    "minibatch_size": 128,
    "learning_rate_decay": 0.995,
    "keep_rate": 0.6,
    "input_swap_noise": 0.05,
    "noise_std": 0.0,
    "n_epochs": 150 # 1000
}

in_dim = 221 # dimension of tfrecord features
BATCH_SIZE = nn_params['minibatch_size']
N_EPOCHS = nn_params['n_epochs']
learning_rate_decay = nn_params['learning_rate_decay']

def _parse_function(record):
    keys_to_features = {
        "rankgauss_feature": tf.FixedLenFeature((221,), tf.float32, default_value=tf.zeros((221,), dtype=tf.float32)),
        "label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)
    # return parsed_features["rankgauss_feature"]
    return parsed_features["rankgauss_feature"], parsed_features["label"]

filenames = [
    os.path.join(data_path, 'train_rankgauss.tfrecord'),
    # os.path.join(data_path, 'test_rankgauss.tfrecord')
]

with tf.variable_scope("supervised_nn_dataset"):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=500000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat(N_EPOCHS)
    iterator = dataset.make_one_shot_iterator()
    next_feature, next_label = iterator.get_next()

# plhd1 = tf.placeholder(tf.float32, [None, 221])

sess = tf.Session()
saver = tf.train.import_meta_graph(stored_model_meta, input_map={"denoise_autoencoder/add:0": next_feature})
# saver = tf.train.import_meta_graph(stored_model_meta, input_map={"denoise_autoencoder/add:0": plhd1})
saver.restore(sess,tf.train.latest_checkpoint(stored_model_path))
graph = tf.get_default_graph()
dae_h1 = graph.get_tensor_by_name("denoise_autoencoder/add_1:0")
dae_h2 = graph.get_tensor_by_name("denoise_autoencoder/add_2:0")
dae_h3 = graph.get_tensor_by_name("denoise_autoencoder/add_3:0")
traing_phase = graph.get_tensor_by_name('PlaceholderWithDefault_3:0')
concat_x = tf.concat([dae_h1, dae_h2, dae_h3], axis=1)


def gaussian_noise_layer(input_layer, std, name):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32, name=name)
    return input_layer + noise
def standard_layer(input_layer, n_nodes, std, keep_rate, bn_phase, layer_name, vscope_name=None):
    # with tf.variable_scope(vscope_name): # you can use scope on outside wrapper
    layer_o = tf.layers.dense(input_layer, n_nodes, name='{}_dense'.format(layer_name))
    layer_o = tf.layers.batch_normalization(layer_o, name='{}_bn'.format(layer_name), training=bn_phase)
    layer_o = tf.nn.relu(layer_o, name='{}_relu'.format(layer_name))
    layer_o = tf.nn.dropout(layer_o, keep_rate, name='{}_dropout'.format(layer_name))
    layer_o = gaussian_noise_layer(layer_o, std, name='{}_gn'.format(layer_name))
    return layer_o

keep_rate = tf.placeholder_with_default(1.0, shape=())
input_swap_noise = tf.placeholder_with_default(0.0, shape=())
noise_std = tf.placeholder_with_default(0.0, shape=())
bn_phase = tf.placeholder_with_default(False, shape=()) # True for train, False for test(emmm...#TODO)
# tf_x = tf.placeholder(tf.float32, [None, in_dim]) #  X_train.shape
# tf_y = tf.placeholder(tf.int32, None)

with tf.variable_scope('supervised_nn_layers'):
    # layer1 = standard_layer(tf_x, nn_params["layers"][0], noise_std, keep_rate, bn_phase, 'layer1')
    layer1 = standard_layer(concat_x, nn_params["layers"][0], noise_std, keep_rate, bn_phase, 'layer1')
    layer2 = standard_layer(layer1, nn_params["layers"][1], noise_std, keep_rate, bn_phase, 'layer2')
    output = tf.layers.dense(layer2, 2)                     # output layer
    global_step = tf.Variable(0, trainable=False)

y_pred_v = tf.nn.softmax(output)
loss = tf.losses.sparse_softmax_cross_entropy(labels=next_label, logits=output)

initial_learning_rate = nn_params['learning_rate'] #初始学习率
learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=11625, # 11625 steps * 128 per batch
                                           decay_rate=learning_rate_decay)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
add_global = global_step.assign_add(1)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops+[add_global]):
    train_op = optimizer.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='supervised_nn_layers'))

summary_l = []
summary_l.append(tf.summary.scalar('nn_loss', loss))
summary_l.append(tf.summary.histogram('nn_layer1', layer1))
summary_l.append(tf.summary.histogram('nn_layer2', layer2))
merge_op = tf.summary.merge(summary_l)


# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
init_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='supervised_nn_layers')
init_op = tf.variables_initializer(var_list=init_var_list, name="supervised_init")
sess.run(init_op)     # initialize var in graph

writer = tf.summary.FileWriter( os.path.join(model_path, 'log'), sess.graph)     # write to file



# x1,x2,x3 = sess.run([dae_h1, dae_h2, dae_h3], feed_dict={plhd1: np.ones([5, 221]), traing_phase: False})
# print('input all ones:',x1.max(),x1.min(),x2.max(),x2.min(),x3.max(),x3.min())
# x1,x2,x3 = sess.run([dae_h1, dae_h2, dae_h3], feed_dict={plhd1: np.zeros([5, 221]), traing_phase: False})
# print('input all zeros:',x1.max(),x1.min(),x2.max(),x2.min(),x3.max(),x3.min())
# without sess.run(init_op)
# input all ones: 7.70159 0.0 6.26328 0.0 4.53865 0.0
# input all zeros: 2.24602 0.0 1.43028 0.0 0.982921 0.0


all_stage = 0
while True:
    try:
        result, loss_step, _, lr = sess.run([merge_op, loss, train_op, learning_rate],
                                {traing_phase: False,
                                 keep_rate: nn_params['keep_rate'],
                                 input_swap_noise: nn_params['input_swap_noise'],
                                 noise_std: nn_params['noise_std'],
                                 bn_phase: True})
        # loss_step, _, lr = sess.run([loss, train_op, learning_rate],
        #                         {traing_phase: False,
        #                          keep_rate: nn_params['keep_rate'],
        #                          input_swap_noise: nn_params['input_swap_noise'],
        #                          noise_std: nn_params['noise_std'],
        #                          bn_phase: True})
        all_stage += 1
        if all_stage % 100 == 0:
            writer.add_summary(result, all_stage)
            print(f'lr {lr:.8f}, EPOCH {all_stage//11625}, all_stage {all_stage}, logloss {loss_step:.8f}')
    except tf.errors.OutOfRangeError:
        break
# %% save tf model, i should save model rather than hidden data
saver = tf.train.Saver()
saver.save(sess, os.path.join(model_path, 'dae_model'))
