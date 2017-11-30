import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data
import mnist_reader
import numpy as np
import matplotlib.pyplot as plt

#加载mnist数据
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)     

X_test, y_test = mnist_reader.load_mnist('../../data/fashion', kind='t10k')
X_train, y_train = mnist_reader.load_mnist('../../data/fashion', kind='train')

HEIGHT = 28
WIDTH = 28

filename_queue = tf.train.string_input_producer(['fashion_mnist.tfrecords'])
reader = tf.TFRecordReader()

_, example = reader.read(filename_queue)
features = tf.parse_single_example(
    example,
    features={
        'label':tf.FixedLenFeature([], tf.int64),
        'image_raw':tf.FixedLenFeature([], tf.string)
    }
)
image = tf.decode_raw(features['image_raw'], tf.uint8) 
image = tf.reshape(image, [28, 28, 1])

label = tf.cast(features['label'], tf.int32)  

image_batch, label_batch = tf.train.batch([image, label],  
                                                  batch_size=1,  
                                                  capacity=10,  
                                                  num_threads=2)  
image = tf.reshape(image_batch, [HEIGHT*WIDTH])


#输入的数据和标签的占位符
x = tf.placeholder(tf.float32, [None, 784])
y_labels = tf.placeholder(tf.float32, shape=[None, 10])


#初始化权重
def weight_var(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
#初始化偏差
def bias_var(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#卷积
def convolution(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#最大池化
def maxPool2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#进行卷积

x_image = tf.reshape(x, [-1, 28, 28, 1])
plt
##一层卷积
Wconvo1 = weight_var([5, 5, 1, 32])
bconvo1 = bias_var([32])

hconvo1 = tf.nn.relu(convolution(x_image, Wconvo1) + bconvo1)
hpool1 = maxPool2x2(hconvo1)
##二层卷积
Wconvo2 = weight_var([5, 5, 32, 64])
bconvo2 = bias_var([64])

hconvo2 = tf.nn.relu(convolution(hpool1, Wconvo2) + bconvo2)
hpool2 = maxPool2x2(hconvo2)

#密集连接层
Wfc1 = weight_var([7 * 7 * 64, 1024])
bfc1 = bias_var([1024])

hpool2flat = tf.reshape(hpool2, [-1, 7*7*64])
hfc1 = tf.nn.relu(tf.matmul(hpool2flat, Wfc1) + bfc1)

#防止过拟合操作,dropout(随机选取权重进行运算)
keep_prob = tf.placeholder("float")
hfc1drop = tf.nn.dropout(hfc1, keep_prob)

#output layer
Wfc2 = weight_var([1024, 10])
bfc2 = bias_var([10])


y_conv = tf.nn.softmax(tf.matmul(hfc1drop, Wfc2) + bfc2)
cross_entropy = -tf.reduce_sum(y_labels*tf.log(y_conv))
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_labels, 1))
#准确度计算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# with tf.name_scope('train'):
#梯度下降
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(200):
        # batch = mnist.train.next_batch(50)
        train_img_label = (np.zeros((50,784)),np.zeros((50,10)))
        for j in range(50):
            img, labelIndex = sess.run([image, label_batch])
            train_img_label[1][j][labelIndex[0]] = 1
            train_img_label[0][j] = img
        if i%10 == 0:
            trainAccuracy = accuracy.eval(feed_dict={
                x: train_img_label[0], y_labels: train_img_label[1], keep_prob: 1.0
            })
            print("[step %d] , [训练正确率]:%g" % (i, trainAccuracy))
        train_step.run(feed_dict={x: train_img_label[0], y_labels: train_img_label[1], keep_prob: 0.5})
    coord.request_stop()
    coord.join(threads)