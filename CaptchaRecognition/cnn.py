import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

# the number of sample
numSam = 1

# 标签转换
def extend(num):
    reli = []
    numStr = str(num)
    for i in range(len(numStr)):
        a = [0]*11
        a[int(numStr[i])] = 1
        reli.append(a)
    return reli


# 初始化权重
def weight_var(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
# 初始化偏差
def bias_var(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积
def convolution(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# 最大池化
def maxPool2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 输入的数据和标签的占位符
x = tf.placeholder(tf.float32, [None, 1600])
y_labels = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 40, 40, 1])

# 一层卷积
Wconvo1 = weight_var([5, 5, 1, 32])
bconvo1 = bias_var([32])

hconvo1 = tf.nn.relu(convolution(x_image, Wconvo1) + bconvo1)
hpool1 = maxPool2x2(hconvo1)

# 二层卷积
Wconvo2 = weight_var([5, 5, 32, 64])
bconvo2 = bias_var([64])

hconvo2 = tf.nn.relu(convolution(hpool1, Wconvo2) + bconvo2)
hpool2 = maxPool2x2(hconvo2)

# 密集连接层
Wfc1 = weight_var([10 * 10 * 64, 1024])
bfc1 = bias_var([1024])

hpool2flat = tf.reshape(hpool2, [-1, 10*10*64])
hfc1 = tf.nn.relu(tf.matmul(hpool2flat, Wfc1) + bfc1)

# 防止过拟合操作,dropout(随机选取权重进行运算)
keep_prob = tf.placeholder("float")
hfc1drop = tf.nn.dropout(hfc1, keep_prob)

# output layer
Wfc2 = weight_var([1024, 10])
bfc2 = bias_var([10])

y_conv = tf.nn.softmax(tf.matmul(hfc1drop, Wfc2) + bfc2)

# yReshape = [tf.slice(y_conv, [0,n,0], [4,1,11]) for n in range(numSam)] # 2指的是batch，n指的是
# y_copy = tf.reshape(yReshape, [numSam,4,11])


# 交叉熵
cross_entropy = -tf.reduce_sum(y_labels*tf.log(y_conv))

# 检查预测值是否正确
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_labels, 1))

# 准确度计算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 梯度下降
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 将文件读取为一个队列
filename_queue = tf.train.string_input_producer(['TrainingSetTwo.tfrecords'])
reader = tf.TFRecordReader()

_, example = reader.read(filename_queue)
features = tf.parse_single_example(
    example,
    features={
        'label':tf.FixedLenFeature([], tf.int64),
        'image_raw':tf.FixedLenFeature([], tf.string)
    }
)
# 还原图像和标签        
label = tf.cast(features['label'], tf.int32) 
image = tf.decode_raw(features['image_raw'], tf.uint8) 
image = tf.reshape(image, [40, 40, 1])
# 打乱图像顺序
image_batch, label_batch = tf.train.shuffle_batch([image, label],  
                                                  batch_size=numSam,  
                                                  capacity=20,  
                                                  num_threads=2,
                                                  min_after_dequeue=10)  
image = tf.reshape(image_batch, (numSam, 40, 40))
# 保存模型
saver = tf.train.Saver() 

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #读取并使用模型
    # ckpt = tf.train.get_checkpoint_state('./model/1')
    # saver.restore(sess, ckpt.model_checkpoint_path)

    for i in range(2000):
        img, label = sess.run([image, label_batch])
        labels = extend(label[0])
        img = imgcrop(img[0])
        img = img / 255
        if i%100 == 0:
            trainAccuracy = accuracy.eval(feed_dict={
                x: img, y_labels: labels, keep_prob: 1.0
            })
            print("[step %d] , [训练正确率]:%g" % (i, trainAccuracy))
            # 模型保存
            saver.save(
                sess, "%s/%s" % ('./model/2', 'model'), global_step=1
            )
        print(sess.run(tf.argmax(y_conv,1), feed_dict={x: img, y_labels: labels, keep_prob: 1.0}))
        train_step.run(feed_dict={x: img, y_labels: labels, keep_prob: 0.5})
    coord.request_stop()
    coord.join(threads)