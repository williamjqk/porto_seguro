# %% this idea is from https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
base_path = '/home/tom/mywork/some_test/porto_seguro/input/'

model_name = 'porto_seguro_dae021'
nn_params = {
    "layers": [1000, 1000],
    "learning_rate": 1e-3, #1e-3, # 3e-3,
    "minibatch_size": 128,
    "learning_rate_decay": 0.995,
    "keep_rate": 0.8,
    "input_swap_noise": 0.05,
    "noise_std": 0.0,
    "n_epochs": 300 # 1000
}

data_path = '/home/tom/data/porto_seguro_dae'
model_path = os.path.join(data_path, model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)

stored_model_name = 'porto_seguro_dae017'
stored_model_meta = os.path.join(data_path, stored_model_name, 'dae_model.meta')
stored_model_path = os.path.join(data_path, stored_model_name)



# %% 创建整个supervised_nn和dataset，载入saver model并重新连接新的dataset和supervised_nn
import tensorflow as tf

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

filenames1 = [
    os.path.join(data_path, 'train_rankgauss_porto_seguro_dae00x3.tfrecord'),
    # os.path.join(data_path, 'test_rankgauss.tfrecord')
]

filenames2 = [
    os.path.join(data_path, 'valid_rankgauss_porto_seguro_dae00x3.tfrecord'),
]

with tf.variable_scope("supervised_nn_dataset"):
    train_dataset = tf.data.TFRecordDataset(filenames1)
    train_dataset = train_dataset.map(_parse_function)
    train_dataset = train_dataset.shuffle(buffer_size=400000)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.repeat(N_EPOCHS)
    train_iterator = train_dataset.make_one_shot_iterator()
    # next_feature, next_label = iterator.get_next()

    valid_dataset = tf.data.TFRecordDataset(filenames2)
    valid_dataset = valid_dataset.map(_parse_function)
    # valid_dataset = valid_dataset.shuffle(buffer_size=100000)
    valid_dataset = valid_dataset.batch(20000) # BATCH_SIZE
    valid_dataset = valid_dataset.repeat(1)
    valid_iterator = valid_dataset.make_initializable_iterator()
    # next_feature, next_label = iterator.get_next()

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    next_feature,next_label = iterator.get_next()



sess = tf.Session()

train_handle = sess.run(train_iterator.string_handle())
valid_handle = sess.run(valid_iterator.string_handle())


saver = tf.train.import_meta_graph(stored_model_meta, input_map={"denoise_autoencoder/add:0": next_feature})
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
# optimizer = tf.train.RMSPropOptimizer(learning_rate)
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

writer = tf.summary.FileWriter( os.path.join(model_path, 'log/train'), sess.graph)     # write to file

valid_gini = tf.placeholder(tf.float32, None, name='valid_gini')
valid_summary_l = []
valid_summary_l.append(tf.summary.scalar('gini', valid_gini))
valid_writer = tf.summary.FileWriter( os.path.join(model_path, 'log/valid'), sess.graph)
valid_merge_op = tf.summary.merge(valid_summary_l)

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
def gini_mlp(pred, y):
    return gini(y, pred) / gini(y, y)

all_stage = 0
while True:
    try:
        result, loss_step, _, lr = sess.run([merge_op, loss, train_op, learning_rate],
                                {handle: train_handle,
                                 traing_phase: False,
                                 keep_rate: nn_params['keep_rate'],
                                 input_swap_noise: nn_params['input_swap_noise'],
                                 noise_std: nn_params['noise_std'],
                                 bn_phase: True})
        all_stage += 1
        if all_stage % 100 == 0:
            writer.add_summary(result, all_stage)
            print(f'lr {lr:.8f}, EPOCH {all_stage//11625}, all_stage {all_stage}, logloss {loss_step:.8f}')
        if all_stage % 2000 == 0:
            y_valid_pred = np.empty(0)
            y_valid_ref = np.empty(0)
            sess.run(valid_iterator.initializer)
            step2 = 0
            while True:
                try:
                    y_pred_v_result, next_label_result = sess.run([y_pred_v, next_label],
                                        {handle: valid_handle,
                                         traing_phase: False,
                                         bn_phase: False})
                    y_valid_pred = np.hstack((y_valid_pred, y_pred_v_result[:,1]))
                    y_valid_ref = np.hstack((y_valid_ref, next_label_result))
                except tf.errors.OutOfRangeError:
                    valid_gini_result = gini_mlp(y_valid_pred, y_valid_ref)
                    valid_merge_op_result = sess.run(valid_merge_op, feed_dict={valid_gini: valid_gini_result})
                    valid_writer.add_summary(valid_merge_op_result, all_stage)
                    print(f"all_stage {all_stage}, write valid_gini score {valid_gini_result}")
                    break
    except tf.errors.OutOfRangeError:
        break
# %% save tf model, i should save model rather than hidden data
saver = tf.train.Saver()
saver.save(sess, os.path.join(model_path, 'dae_model'))
