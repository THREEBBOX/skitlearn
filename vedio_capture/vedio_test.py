# -*- coding: utf-8 -*-
# @Time    : 2020/1/23 16:16
# @Author  : WangXY TBX
# @File    : vedio_test.py
# @Software: PyCharm
# @Description: Designed by TBX WANG
import cv2;
import numpy as np
import random as rd
if __name__ == '__main__':
    count=1
    cap=cv2.VideoCapture(0)
    flag=True
    while True:
        sucess,img=cap.read()
        fps=cap.get(cv2.CAP_PROP_FPS)
        fps+=rd.randint(1,10)

        # img = cv2.rectangle(img, (200, 200), (630 , 400), (255, 0, 0), 2)
        img=cv2.line(img,(280,240),(360,240),(0,0,255),thickness=5)
        img=cv2.line(img,(320,200),(320,280),(0,0,255),thickness=5)
        cv2.putText(img,"fps:"+str(fps),(560,40),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,0,255),1)
        if flag:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(fps)

        cv2.imshow("video capture test",img)
        k=cv2.waitKey(1)
        if k==27:
            cv2.destroyAllWindows()
            break
        if k==ord("s"):
            flag=not flag
    cap.release()

