import cv2 # 导入opencv库
import os

import numpy as np
#cv2.namedWindow("1")
#cv2.namedWindow("2")
#cv2.namedWindow("3")
#cv2.namedWindow("4")
def Pretreatment( imgpath ):
    img=cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
    img1=cv2.GaussianBlur(img,(0,0),1)
    #thresh1 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,19,11)
    m = np.reshape(img1, [1, 512*512])
    mean = m.sum() / (512*512)
    ret,thresh1 = cv2.threshold(img1,mean,255,cv2.THRESH_BINARY)
    #ret,thresh2 = cv2.threshold(img1,40,255,cv2.THRESH_BINARY)
    cv2.imshow("1",img1)
    cv2.imshow("2",thresh1)
    contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(thresh1,contours,-1,(0,0,255),3)
    w=1
    h=1
    ii=0
    while w*h<100000 and ii<45:
        x, y, w, h = cv2.boundingRect(contours[i])
        ii=ii+1
    cropImg = img1[y:y+h,x:x+w]
    pic = cv2.resize(cropImg, (512, 512), interpolation=cv2.INTER_LINEAR)
    #cropImg1 = thresh1[y:y+h,x:x+w]
    #pic1 = cv2.resize(cropImg1, (512, 512), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(imgpath, pic)
    return pic
#cv2.imwrite(imagepath + path[j] + "/ex/" + image_name[i],pic)
#cv2.imshow("1",cropImg)

#cv2.waitKey(0)


#cv2.destroyAllWindows()