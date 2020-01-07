# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 18:55
# @Author  : WangXY TBX
# @File    : test.py
# @Software: PyCharm
# @Description: Designed by TBX WANG
import cv2
import matplotlib.pyplot as plt



max_output_value = 255
mytest=cv2.imread("skitlearn/PRECESSING/ImageData/mytest.png",cv2.IMREAD_GRAYSCALE)
neighbour_size = 35
subtract_from_mean = 5
image_binary=cv2.adaptiveThreshold(mytest,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,neighbour_size,subtract_from_mean)
plt.imshow(image_binary,cmap='gray')
plt.show(image_binary.all())
cv2.imwrite("skitlearn/PRECESSING/ImageData/testresult.jpeg",image_binary)