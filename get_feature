#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#get_feature
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('img/1.bmp')
crop = img[149:174,263:282,0]
crop = cv2.resize(crop,(18,27))
plt.imshow(crop,cmap = 'gray')

def get_feature(img):
    img = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 10)
    img[np.where(img==0)]=1
    img[np.where(img==255)]=0
    
    feature_count = []
    feature_difference_count = []
    temp = []
    for j in range(3):
        for k in range(3):
            temp.append(int(np.sum(img[9*j:9*j+9,6*k:6*k+6])))
    feature_count.append(temp)
    feature_difference_count.append([temp[1]-temp[0],temp[2]-temp[1],temp[4]-temp[3],
                     temp[5]-temp[4],temp[7]-temp[6],temp[8]-temp[7],
                     temp[3]-temp[0],temp[6]-temp[3],temp[4]-temp[1],
                     temp[7]-temp[4],temp[5]-temp[2],temp[8]-temp[5]])
    feature = []
    feature.append(feature_count)
    feature.append(feature_difference_count)
    return feature

feature = get_feature(crop)

match_to_output = ['0','1','2','3','4','5','6','7','8','9',
                   'a','b','c','d','e','f','g','h','i','j','k','l','m','n',
                   'o','p','q','r','s','t','u','v','w','x','y','z',
                   'A','B','C','D','E','F','G','H','I','J','K','L','M','N',
                   'O','P','Q','R','S','T','U','V','W','X','Y','Z']
#match
def count_mean_square(list1,list2):
    count = 0
    for i,j in zip(list1,list2):
        count+= (i-j)**2
    return count
feature_all = []
feature_all_difference = []  

def get_match_num(feature):
    feature_count,feature_difference_count = feature[0],feature[1]
    output = []
    output_difference = []
    for i in range(len(62)):
        output.append(count_mean_square(feature_count,feature_all[i]))
        output_difference.append(count_mean_square(feature_difference_count,feature_all_difference[i]))
    match_index =   [i for i,v in sorted(enumerate(output), key=lambda x:x[1],reverse=True)][0]   
    match_difference_index = [i for i,v in sorted(enumerate(output_difference), key=lambda x:x[1],reverse=True)][0]
    return match_index
        
        
    
    



