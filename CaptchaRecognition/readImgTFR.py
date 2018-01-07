import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pprint import pprint

# 标签转换
def extendNum(num,p=1.1,ind=0):
    reList = [0]*11
    numStr = str(num)
    for i in numStr:
        reList[int(i)] = p
        p -= 0.1
    return reList

#将文件读取为一个队列
filename_queue = tf.train.string_input_producer(['TrainingSet1.tfrecords'])
reader = tf.TFRecordReader()

_, example = reader.read(filename_queue)
features = tf.parse_single_example(
    example,
    features={
        'label':tf.FixedLenFeature([], tf.int64),
        'image_raw':tf.FixedLenFeature([], tf.string)
    }
)
#还原图像和标签        
label = tf.cast(features['label'], tf.int32) 
image = tf.decode_raw(features['image_raw'], tf.uint8) 
image = tf.reshape(image, [40, 40, 1])
# #打乱图像顺序
image_batch, label_batch = tf.train.shuffle_batch([image, label],  
                                                  batch_size=1,  
                                                  capacity=20,  
                                                  num_threads=2,
                                                  min_after_dequeue=10)  
image = tf.reshape(image_batch, (1, 40, 40))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    percent = [0]*4
    ##可视化tfrecords中的图像
    for i in range(100):
        img, label = sess.run([image, label_batch])
        if i%100 == 0:
            print('——[进行到第%d步]——' % (i))
        if 10 <= label < 100:
            percent[1] += 1
        elif 100 <= label < 1000:
            percent[2] += 1
        elif 1000 <= label:
            percent[3] += 1
        else:
            percent[0] += 1
        print(label)
    print(percent)
    numSum = sum(percent)
    for j in range(4):
        print('[%d位数验证码在数据集中所占比例为: %f]' % (j+1, percent[j]/numSum))
    coord.request_stop()
    coord.join(threads)