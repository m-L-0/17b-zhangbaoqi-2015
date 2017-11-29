import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from math import *

#将文件读取为一个队列
filename_queue = tf.train.string_input_producer(['./zh_character_test.tfrecords'])
reader = tf.TFRecordReader()

_, example = reader.read(filename_queue)
features = tf.parse_single_example(
    example,
    features={
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'image_raw':tf.FixedLenFeature([], tf.string)
    }
)

#还原图像和标签
label = tf.cast(features['label'], tf.int32)  
height = tf.cast(features['height'], tf.int32)  
image = tf.decode_raw(features['image_raw'], tf.uint8) 


#打乱图像顺序
# image_batch, label_batch = tf.train.batch([image, label], batch_size=1,num_threads=2,capacity=30)
# image_batch, label_batch,height_batch = tf.train.shuffle_batch([image, label, height],  
#                                                 batch_size=1,  
#                                                 capacity=100,  
#                                                 num_threads=2,
#                                                 min_after_dequeue=50)  
# image = tf.reshape(image_batch, (height_batch, -1))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    ##可视化tfrecords中的图像
    for i in range(3):
        img, labels, h = sess.run([image, label, height])
        # img = img.reshape(h, -1)
        print(labels)
        plt.imshow(img)
        plt.show()
    coord.request_stop()
    coord.join(threads)