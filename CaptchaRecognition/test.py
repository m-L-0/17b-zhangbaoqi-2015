from pprint import pprint
import numpy as np

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

def calculen(num,i=10):
    rl = [0]*4
    rl[len(str(num))-1] = 1
    return rl
l = [641, 1024, 234,532,5432,678,1111]
def imgcrop(imgl):
    img = Image.fromarray(imgl)
    im1 = img.crop(0,0,20,40)
    im2 = img.crop(20,0,40,40)
    return im1,im2

print(extendNum(764))