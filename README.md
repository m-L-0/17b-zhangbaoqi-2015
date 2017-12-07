# 17b-zhangbaoqi-2015
machine-learning works

### 第一次作业实现-FashionMNIST_Challenge
* 根据要求将图片数据保存成`.tfrecords`格式文件  by [CreatTFR](https://github.com/m-L-0/17b-zhangbaoqi-2015/blob/master/CreatTFR.ipynb)
* 读取`.tfrecords`格式文件存储的图片数据，并且以队列的形式读取并可视化  by [ReadTFR](https://github.com/m-L-0/17b-zhangbaoqi-2015/blob/master/ReakTFR.ipynb)
* 算法实现:
  - 利用KNN算法对图片进行分类  by [KNN](https://github.com/m-L-0/17b-zhangbaoqi-2015/blob/master/MNIST-Algorithm-code/KNN.ipynb)
  - 利用K-means算法进行聚类   by [K-means](https://github.com/m-L-0/17b-zhangbaoqi-2015/blob/master/MNIST-Algorithm-code/K-Means.ipynb)
  - 利用CNN算法进行图片分类   by [CNN](https://github.com/m-L-0/17b-zhangbaoqi-2015/blob/master/MNIST-Algorithm-code/CNN.ipynb) CNN算法的tensorboard可视化  by [CNN_tensorboard](https://github.com/m-L-0/17b-zhangbaoqi-2015/blob/master/MNIST-Algorithm-code/CNN_tensorflow.ipynb)

#### 算法实现过程中用到的数学知识：  
1. 欧氏距离
2. 梯度下降的相关数学实现

### 第二次作业实现-Vehicle_License_Plate_Recognition
* 将已分类好的图片存储到TFrecords文件中     by [CreatTFrecords](https://github.com/m-L-0/17b-zhangbaoqi-2015/blob/master/License_plate_Recognition-Algorithm-code/CreatTFrecords.ipynb)
* 读取文TFrecords文件，对数据进行处理       by [ReadTFrecords](https://github.com/m-L-0/17b-zhangbaoqi-2015/blob/master/License_plate_Recognition-Algorithm-code/ReadTFrecords.ipynb)
* 算法实现：
  - 使用CNN算法对图片进行训练，并保存模型    by [CNNTrain](https://github.com/m-L-0/17b-zhangbaoqi-2015/blob/master/License_plate_Recognition-Algorithm-code/cnnTrain.ipynb)
* 使用保存的模型对验证集图片进行分类，并计算正确率和召回率    by [CNNVerify](https://github.com/m-L-0/17b-zhangbaoqi-2015/blob/master/License_plate_Recognition-Algorithm-code/cnnVerify.ipynb)
* 文档说明    by [documents-description](https://github.com/m-L-0/17b-zhangbaoqi-2015/blob/master/License_plate_Recognition-Algorithm-code/documents-description.md)
