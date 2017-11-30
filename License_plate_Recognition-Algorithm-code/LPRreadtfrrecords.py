import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from math import *
from PIL import Image

#将文件读取为一个队列
filename_queue = tf.train.string_input_producer(['./zh_character.tfrecords'])
reader = tf.TFRecordReader()

_, example = reader.read(filename_queue)
features = tf.parse_single_example(
    example,
    features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw':tf.FixedLenFeature([], tf.string)
    }
)

#还原图像和标签 
image = tf.decode_raw(features['image_raw'], tf.uint8) 
image = tf.reshape(image, [30, 30, 1])
label = tf.cast(features['label'], tf.int32) 

#打乱图像顺序
image_batch, label_batch = tf.train.shuffle_batch([image, label],  
                                                batch_size=1,  
                                                 capacity=200,  
                                                num_threads=2,
                                                min_after_dequeue=100)  
image = tf.reshape(image_batch, (30, 30))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    ##可视化tfrecords中的图像
    for i in range(5):
        img, labels = sess.run([image, label_batch])
        print(labels)
        #黑白反向
        im = Image.fromarray(255-img)
        im.show()
        i = np.array(im)
        plt.imshow(i)
        plt.show()
    coord.request_stop()
    coord.join(threads)