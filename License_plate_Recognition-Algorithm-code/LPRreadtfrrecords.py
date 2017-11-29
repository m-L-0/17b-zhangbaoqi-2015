import tensorflow as tf
import skimage.io as io
import matplotlib.pyplot as plt
from math import *

HEIGHT = 28
WIDTH = 28

#将文件读取为一个队列
filename_queue = tf.train.string_input_producer(['./zh_character.tfrecords'])
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
image = tf.decode_raw(features['image_raw'], tf.uint8) 
image = tf.reshape(image, [28, 28, 1])
label = tf.cast(features['label'], tf.int32)  

#打乱图像顺序
image_batch, label_batch = tf.train.shuffle_batch([image, label],  
                                                  batch_size=1,  
                                                  capacity=100,  
                                                  num_threads=2,
                                                  min_after_dequeue=50)  
image = tf.reshape(image_batch, (HEIGHT, WIDTH))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    c = 0

    img_label_list = []
    coordinateSet = []
    o_distance_list = []
    ##可视化tfrecords中的图像
    for i in range(3):
        img, label = sess.run([image, label_batch])
        print("[output]:",label)
        plt.imshow(img)
        plt.show()
        img_label_list.append((img, label))
    
    coord.request_stop()
    coord.join(threads)