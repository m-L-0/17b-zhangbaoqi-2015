from PIL import Image
from pprint import pprint
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import re

# 标签长度分类
def calculen(num,i=10):
    return len(str(num))

#转换存储类型为bytes
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#转换存储特征类型为int64
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 正则表达式提取字符串中的数字
restr = lambda s: re.findall(r'\d+', s)

# 读取标签csv表
labelPath = '../data/captcha/labels/'
f = open(labelPath + 'labels.csv', 'r')
listla = f.readlines()
imgLabel = [restr(s) for s in listla]
imLaDict = dict(imgLabel)

# 读取图片
imgPath = '../data/captcha/images/'
listd = os.listdir(imgPath)
i = 1
c = 0
for n in range(len(listd)):
    img = Image.open(imgPath + listd[n])
    img = img.resize((40, 40))
    img = np.array(img.convert('1'))
    imgIndex = restr(listd[n])[0]
    label = imLaDict[imgIndex]
    #将图像转化为二进制形式
    img_raw = img.tobytes()
    
    if n == 0:
        writer = tf.python_io.TFRecordWriter("./TrainingSet"+ str(i) +".tfrecords")
    if n != 0 and n%5000 == 0 and n < 32001:
        i += 1
        print('————create————')
        writer.close()
        writer = tf.python_io.TFRecordWriter("./TrainingSet"+ str(i) +".tfrecords")
    if n == 32001:
        writer.close()
        print('————create————')
        writer = tf.python_io.TFRecordWriter("./TestSet.tfrecords")
    if n == 36001:
        writer.close()
        print('————create————')
        writer = tf.python_io.TFRecordWriter("./VerifySet.tfrecords")

    #tfrecords文件存储特征值
    if calculen(label) == 2:
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(label)),
            'image_raw': _bytes_feature(img_raw)
        }))
        writer.write(example.SerializeToString())
        print('write', c)
    print(c)
    c += 1
print("create tfrecords file successful!")
writer.close()
f.close()