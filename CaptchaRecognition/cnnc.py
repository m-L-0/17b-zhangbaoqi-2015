import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

# the number of sample
numSam = 100
imgshape=1600
outlen=4

labelsDict = {k:v for k,v in zip(range(10),range(10))}
labelsDict[10] = ' '

# 标签转换
def extendNum(num):
    reList = [[0]*11 for i in range(4)]
    numStr = str(num)
    numStr += 'n'*(4-len(numStr))
    for i in range(len(numStr)):
        if numStr[i] == 'n':
            reList[i][10] = 1
        else:
            reList[i][int(numStr[i])] = 1
    return reList

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

# 卷积
def cnn(imgshape, switch=False, sess=tf.Session()):

    #输入的数据和标签的占位符
    x = tf.placeholder(tf.float32, [None, imgshape])
    y_labels = tf.placeholder(tf.float32, shape=[None, 4, 11])

    x_image = tf.reshape(x, [-1, 40, 40, 1])

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
    Wfc1 = weight_var([10 * 10 * 64, 1024])
    bfc1 = bias_var([1024])

    hpool2flat = tf.reshape(hpool2, [-1, 10*10*64])
    hfc1 = tf.nn.relu(tf.matmul(hpool2flat, Wfc1) + bfc1)

    #防止过拟合操作,dropout(随机选取权重进行运算)
    keep_prob = tf.placeholder("float")
    hfc1drop = tf.nn.dropout(hfc1, keep_prob)

    #output layer
    with tf.name_scope('weight'):
        Wfc2 = [weight_var([1024, 11]) for i in range(4)]
    # tf.histogram_summary("weights",Wfc2)
    with tf.name_scope('bias'):
        bfc2 = [bias_var([11]) for i in range(4)]

    # y_conv = tf.nn.softmax(tf.matmul(hfc1drop, Wfc2) + bfc2)
    y_conv = [tf.nn.softmax(tf.matmul(hfc1drop, Wfc2[i]) + bfc2[i]) for i in range(4)]

    # yReshape = [tf.slice(y_conv, [0,n,0], [4,1,11]) for n in range(numSam)] # 2指的是batch，n指的是
    # y_copy = tf.reshape(yReshape, [numSam,4,11])
    yRelabel = [tf.slice(y_labels, [0,n,0], [numSam,1,11]) for n in range(4)] # 2指的是batch，n指的是
    y_cola = tf.reshape(yRelabel, [4,numSam,11])

    #交叉熵
    cross_entropy = [-tf.reduce_sum(y_cola[i]*tf.log(y_conv[i]+1e-8)) for i in range(4)]

    #检查预测值是否正确
    correct_prediction = tf.equal(tf.argmax(y_conv, 2), tf.argmax(y_cola, 2))

    #准确度计算
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    sumCro = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', sumCro)
    #梯度下降
    train_step = tf.train.AdamOptimizer().minimize(sumCro)

    # 模型保存
    EigEm = {
        'x':x,
        'y_labels':y_labels,
        'keep_prob':keep_prob,
        'y_conv':y_conv, 
        'cross_entropy':cross_entropy, 
        'correct_prediction':correct_prediction, 
        'accuracy':accuracy, 
        'train_step':train_step,
    }
    return EigEm


# 将文件读取为一个队列
filename_queue = tf.train.string_input_producer(['TrainingSet3.tfrecords'])
# filename_queue = tf.train.string_input_producer(['TestSet.tfrecords'])
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
# #打乱图像顺序
image_batch, label_batch = tf.train.shuffle_batch([image, label],  
                                                  batch_size=numSam,  
                                                  capacity=20,  
                                                  num_threads=2,
                                                  min_after_dequeue=10)  
image = tf.reshape(image_batch, (numSam, 40*40))
EigEm1 = cnn(imgshape=1600)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 读取并使用模型
    ckpt = tf.train.get_checkpoint_state('./model1')
    saver.restore(sess, ckpt.model_checkpoint_path)

    #变量汇总
    merged = tf.summary.merge_all()
    #写入数据
    summary_writer = tf.summary.FileWriter('graph/', graph=sess.graph)

    for i in range(2000):
        img, label = sess.run([image, label_batch])
        labels = [extendNum(num) for num in label]
        img = img / 255
        if i%10 == 0:
            trainAccuracy = EigEm1['accuracy'].eval(feed_dict={
                EigEm1['x']: img, EigEm1['y_labels']: labels, EigEm1['keep_prob']: 1.0
            })
            print("[step %d] , [训练正确率]:%g" % (i, trainAccuracy))
            summary,_ = sess.run([merged, EigEm1['train_step']], 
                                 feed_dict={EigEm1['x']: img, EigEm1['y_labels']: labels, EigEm1['keep_prob']: 1.0})
            summary_writer.add_summary(summary, i)
        saves = saver.save(
            sess, "%s/%s" % ('./model1/', 'model1'), global_step=1
        )
        # print(sess.run(tf.reduce_mean(EigEm1['cross_entropy']), feed_dict={EigEm1['x']: img, EigEm1['y_labels']: labels, EigEm1['keep_prob']: 1.0}))
        EigEm1['train_step'].run(feed_dict={EigEm1['x']: img, EigEm1['y_labels']: labels, EigEm1['keep_prob']: 0.5})
    saver.save(sess, save_path='graph/', global_step=1)
    coord.request_stop()
    coord.join(threads)