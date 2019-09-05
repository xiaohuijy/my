import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

path_file = glob('produce/*.*')

for each_file in path_file:
    img0 = cv2.imread(each_file)
    img0 = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)    
    img = cv2.adaptiveThreshold(img0,1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 25)
    img = 1-img
    s1= np.sum(img,axis = 1)
    s2 = s1[len(s1)//2-15:len(s1)//2+15]
    for i in range(len(s2)):
        if s2[i]<5:
            st = i
            break
    for i in range(len(s2)-1,0,-1):
        if s2[i]<5:
            end = i
            break
    st+=len(s1)//2-15
    end+=len(s1)//2-15
    mid = (st+end)//2 
    crop_each_file = each_file.replace('produce','produce_crop')
    
    c1 = img0[:mid]
    c1 = cv2.adaptiveThreshold(c1,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 15)
    c2 = img0[mid:]  
    c2 = cv2.adaptiveThreshold(c2,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 15)
    cv2.imwrite(crop_each_file+'_0.jpg',c1)
    cv2.imwrite(crop_each_file+'_1.jpg',c2)
