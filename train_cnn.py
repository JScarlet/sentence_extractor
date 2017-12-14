import tensorflow as tf
import numpy as np
from word_vec_generator import WordVecGenerator

__author__ = 'chapter'

path = "train_data.json"
type_num = 4

def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


word_generator = WordVecGenerator()
data, label = word_generator.data_list_extraction(path)

#打乱顺序
#num_example = data.shape[0]
#arr = np.arange(num_example)
#np.random.shuffle(arr)
#data = data[arr]
#label = label[arr]

print("Download Done!")

sess = tf.InteractiveSession()

# paras
W_conv1 = weight_varible([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# conv layer-1
x = tf.placeholder(tf.float32, [None, 8000], name='x')
x_image = tf.reshape(x, [-1, 20, 400, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# conv layer-2
W_conv2 = weight_varible([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# full connection
W_fc1 = weight_varible([100 * 5 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 100 * 5 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer: softmax
W_fc2 = weight_varible([1024, type_num])
b_fc2 = bias_variable([type_num])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y_conv')
y_ = tf.placeholder(tf.float32, [None, type_num])

# model training
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())

n_epoch = 10
batch_size = 50
for epoch in range(n_epoch):
    train_loss, train_accuracy, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(data, label, batch_size, True):
        _, err, ac = sess.run([train_step, cross_entropy, accuracy], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err
        train_accuracy += ac
        n_batch += 1
    print("train loss: " + (np.sum(train_loss) / n_batch))
    print("train accuracy: " + (np.sum(train_accuracy) / n_batch))

saver.save(sess, 'model/model.ckpt')
sess.close()


# accuacy on test
#print("test accuracy %g"%(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))