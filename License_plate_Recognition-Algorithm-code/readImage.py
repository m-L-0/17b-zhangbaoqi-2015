
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os 

imgSaveList = [[] for i in range(32)]
#读取图片文件
rodir = './image-data/汉字/'
pathlist = os.listdir(rodir)
for pathName in pathlist[30]:
    for imgPathName in os.listdir(rodir+pathName)[:5]:
        img = Image.open(rodir+pathName+'/'+imgPathName)
        img = img.convert('L').resize((30, 30))
        img.show()
        # plt.imshow(img)
        # plt.show()
        imgNparr = np.array(img)
        plt.imshow(imgNparr)
        plt.show()