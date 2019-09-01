# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 16:43:24 2019

@author: Administrator
"""

import xml.etree.ElementTree as ET
import cv2
from glob import glob
import random
import numpy as np

def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum // len(num)

def get_ground(gray, blockSize):
    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]
            imageROI = gray[rowmin:rowmax, colmin:colmax]
            ss= imageROI.flatten().tolist()
            temp = sorted(ss,reverse=True)[1:7]
            #gray[rowmin:rowmax, colmin:colmax]=averagenum(temp)
            for rr in range(rowmin,rowmax):
                for cc in range(colmin,colmax):
                    a_max = averagenum(temp)+15
                    if a_max>255:
                        a_max = 255
                    a_min = averagenum(temp)-15
                    gray[rr][cc] = random.randint(a_min,a_max)
    return gray

def unevenLightCompensate(img, blockSize):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    #dst = cv2.GaussianBlur(dst, (3, 3), 0)
    #dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    return dst

def transform(img):
    temp = img.copy()
    temp = unevenLightCompensate(temp,32)
    roi= temp.flatten().tolist()
    ground_temp = sorted(roi,reverse=True)[1:7]
    ground_temp = averagenum(ground_temp)
    img =  cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 10)
    #img = cv2.dilate(img,np.array([1,1]))
# =============================================================================
#     img[np.where(img==0)]=35
#     img[np.where(img==1)]=120
# =============================================================================
    for m in range(27):
        for n in range(18):
            if img[m][n]==0:
                img[m][n]=temp[m][n]+random.randint(-5,5)

    return img


img_file = glob('./created_from_good_crop/*.*')
for filename in img_file:
    xml_file = filename.replace('created_from_good_crop','created_from_good_crop_label')
    xml_file = xml_file.replace('.bmp','.xml')                
    tree=ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    obj = root.find('object')
    xmlbox = obj.find('bndbox')
    xmin = int(xmlbox.find('xmin').text)
    ymin = int(xmlbox.find('ymin').text)
    xmax = int(xmlbox.find('xmax').text)
    ymax = int(xmlbox.find('ymax').text)
    
    pipei = cv2.imread(filename)
    pipei  =cv2.cvtColor(pipei,cv2.COLOR_BGR2GRAY)
    crop = pipei[ymin:ymax,xmin:xmax]
    
    big_file = glob('./created_pic/big/*.*')
    small_file = glob('./created_pic/small/*.*')
    num_file = glob('./created_pic/num/*.*')
    all_file = []
    all_file.extend(big_file)
    all_file.extend(small_file)
    all_file.extend(num_file)
    dict_all_file = dict(enumerate(all_file))
    random_file_0 = []
    for i in range(9):
        random_file_0.append(dict_all_file[random.randint(0,61)])
    gray1  =cv2.cvtColor(cv2.imread(random_file_0[0]), cv2.COLOR_BGR2GRAY)
    gray1 = transform(gray1)
    gray2  =cv2.cvtColor(cv2.imread(random_file_0[1]), cv2.COLOR_BGR2GRAY)
    gray2 = transform(gray2)
    gray3  =cv2.cvtColor(cv2.imread(random_file_0[2]), cv2.COLOR_BGR2GRAY)
    gray3 = transform(gray3)
    gray4  =cv2.cvtColor(cv2.imread(random_file_0[3]), cv2.COLOR_BGR2GRAY)
    gray4 = transform(gray4)
    gray5  =cv2.cvtColor(cv2.imread(random_file_0[4]), cv2.COLOR_BGR2GRAY)
    gray5 = transform(gray5)
    gray6  =cv2.cvtColor(cv2.imread(random_file_0[5]), cv2.COLOR_BGR2GRAY)
    gray6 = transform(gray6)
    gray7  =cv2.cvtColor(cv2.imread(random_file_0[6]), cv2.COLOR_BGR2GRAY)
    gray7 = transform(gray7)
    gray8  =cv2.cvtColor(cv2.imread(random_file_0[7]), cv2.COLOR_BGR2GRAY)
    gray8 = transform(gray8)
    gray9  =cv2.cvtColor(cv2.imread(random_file_0[8]), cv2.COLOR_BGR2GRAY)
    gray9 = transform(gray9)
    add_pic = np.ones((27,2))
    image_0 = np.concatenate((gray1,add_pic,gray2,add_pic,gray3,add_pic,gray4,add_pic,gray5,add_pic,gray6,add_pic,gray7,add_pic,gray8,add_pic,gray9),axis=1)
    
    random_file_1 = []
    for i in range(9):
        random_file_1.append(dict_all_file[random.randint(0,61)])
    gray1  =cv2.cvtColor(cv2.imread(random_file_1[0]), cv2.COLOR_BGR2GRAY)
    gray1 = transform(gray1)
    gray2  =cv2.cvtColor(cv2.imread(random_file_1[1]), cv2.COLOR_BGR2GRAY)
    gray2 = transform(gray2)
    gray3  =cv2.cvtColor(cv2.imread(random_file_1[2]), cv2.COLOR_BGR2GRAY)
    gray3 = transform(gray3)
    gray4  =cv2.cvtColor(cv2.imread(random_file_1[3]), cv2.COLOR_BGR2GRAY)
    gray4 = transform(gray4)
    gray5  =cv2.cvtColor(cv2.imread(random_file_1[4]), cv2.COLOR_BGR2GRAY)
    gray5 = transform(gray5)
    gray6  =cv2.cvtColor(cv2.imread(random_file_1[5]), cv2.COLOR_BGR2GRAY)
    gray6 = transform(gray6)
    gray7  =cv2.cvtColor(cv2.imread(random_file_1[6]), cv2.COLOR_BGR2GRAY)
    gray7 = transform(gray7)
    gray8  =cv2.cvtColor(cv2.imread(random_file_1[7]), cv2.COLOR_BGR2GRAY)
    gray8 = transform(gray8)
    gray9  =cv2.cvtColor(cv2.imread(random_file_1[8]), cv2.COLOR_BGR2GRAY)
    gray9 = transform(gray9)
    image_1 = np.concatenate((gray1,add_pic,gray2,add_pic,gray3,add_pic,gray4,add_pic,gray5,add_pic,gray6,add_pic,gray7,add_pic,gray8,add_pic,gray9),axis=1)
    image = np.concatenate((image_0,np.ones((4,178)),image_1)).astype(np.uint8)
    
    crop = cv2.resize(crop,(image.shape[1],image.shape[0]))
    ss = get_ground(crop,8) 
    for m in range(58):
        for n in range(178):
            if image[m][n]==1:
                image[m][n]=ss[m][n]
    image = cv2.resize(image,(xmax-xmin,ymax-ymin))
    #image = cv2.GaussianBlur(image,(5,5),2)
    pipei[ymin:ymax,xmin:xmax] = image[:,:]     
    cv2.imwrite(filename.replace('created_from_good_crop','created_good_pic'),pipei)          
                
