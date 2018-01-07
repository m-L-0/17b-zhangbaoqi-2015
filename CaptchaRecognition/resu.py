import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

def cnnmain(img):
    # the number of sample
    numSam = 1
    imgshape=1600
    outlen=4
    fm = 100

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
        Wfc2 = [weight_var([1024, 11]) for i in range(4)]
        bfc2 = [bias_var([11]) for i in range(4)]

        y_conv = [tf.nn.softmax(tf.matmul(hfc1drop, Wfc2[i]) + bfc2[i]) for i in range(4)]

        yRelabel = [tf.slice(y_labels, [0,n,0], [numSam,1,11]) for n in range(4)] 
        y_cola = tf.reshape(yRelabel, [4,numSam,11])

        #交叉熵
        cross_entropy = [-tf.reduce_sum(y_cola[i]*tf.log(y_conv[i]+1e-8)) for i in range(4)]

        #检查预测值是否正确
        correct_prediction = tf.equal(tf.argmax(y_conv, 2), tf.argmax(y_cola, 2))

        #准确度计算
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sumCro = tf.reduce_mean(cross_entropy)
        #梯度下降
        train_step = tf.train.AdamOptimizer().minimize(sumCro)

        saver = tf.train.Saver() 
        # 保存模型
        if switch:
            # 读取并使用模型
            ckpt = tf.train.get_checkpoint_state('./model')
            saver.restore(sess, ckpt.model_checkpoint_path)
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

    EigEm1 = cnn(imgshape=1600)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 读取并使用模型
        ckpt = tf.train.get_checkpoint_state('./model')
        saver.restore(sess, ckpt.model_checkpoint_path)

        img = img / 255
        outlayer = sess.run(tf.argmax(EigEm1['y_conv'], 2), feed_dict={EigEm1['x']: img, EigEm1['keep_prob']: 1.0})
        outStr = ''
        for elem in outlayer:
            outStr += str(labelsDict[elem[0]])
        return outStr
        coord.request_stop()
        coord.join(threads)
