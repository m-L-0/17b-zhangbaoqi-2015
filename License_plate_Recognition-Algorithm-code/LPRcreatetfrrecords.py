from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os 

imgSaveList = [[] for i in range(32)]
#读取图片文件
rodir = './image-data/汉字/'
pathlist = os.listdir(rodir)
ind = 0
for pathName in pathlist:
    for imgPathName in os.listdir(rodir+pathName):
        img = Image.open(rodir+pathName+'/'+imgPathName)
        imgNparr = np.array(img.resize((30,30)))
        imgSaveList[ind].append(imgNparr[:,:,1])
    ind += 1

#转换存储类型为bytes
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#转换存储特征类型为int64
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

writer = tf.python_io.TFRecordWriter("zh_character.tfrecords")

for imgIndex in range(len(imgSaveList)):
    for x_img in imgSaveList[imgIndex]:
        #将图像转化为二进制形式
        img_raw = x_img.tobytes() 
        #存储标签
        label = imgIndex
        
        #tfrecords文件存储特征值
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(img_raw)
        }))
        writer.write(example.SerializeToString())
    print(imgIndex)
print("create tfrecords file successful!")
writer.close()