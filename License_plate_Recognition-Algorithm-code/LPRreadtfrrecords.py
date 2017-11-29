import tensorflow as tf
import matplotlib.pyplot as plt
from math import *

#将文件读取为一个队列
filename_queue = tf.train.string_input_producer(['./zh_character_hw.tfrecords'])
reader = tf.TFRecordReader()

_, example = reader.read(filename_queue)
features = tf.parse_single_example(
    example,
    features={
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw':tf.FixedLenFeature([], tf.string)

    }
)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #还原图像和标签
    label = tf.cast(features['label'], tf.int32)  
    height = tf.cast(features['height'], tf.int64)
    width = tf.cast(features['width'], tf.int64)
    image = tf.decode_raw(features['image_raw'], tf.uint8) 
    image = tf.reshape(image, [height, width, 1])


    #打乱图像顺序
    image_batch, label_batch = tf.train.shuffle_batch([image, label],  
                                                    batch_size=1,  
                                                    capacity=100,  
                                                    num_threads=2,
                                                    min_after_dequeue=50)  
    image = tf.reshape(image_batch, (height, width))

    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    ##可视化tfrecords中的图像
    for i in range(3):
        _, label = sess.run([image, label_batch])
        print(sess([height, width]))
        print("[output]:",label)
    # coord.request_stop()
    # coord.join(threads)