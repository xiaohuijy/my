#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
img = cv2.imread('1.bmp')
c1 = img[149:177,261:426,0]
c2 = img[178:206,261:426,0]

b_c1_ = cv2.adaptiveThreshold(c1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 10)
b_c1 = cv2.erode(b_c1_,kernel = np.ones((3,3)))
#background = np.zeros((28,165))

background = c1.copy()
background[np.where(b_c1==0)]=0
for m in range(28):
    for n in range(165):
        if background[m][n]==0:
            y0 = (m-3 if m-3>=0 else 0)
            y1 = (m+4 if m+4<=28 else 28)
            x0 = (n-3 if n-3>=0 else 0)
            x1 = (n+4 if n+4<=165 else 165)
            kernel = background[y0:y1,x0:x1].flatten()
            kernel1 = np.delete(kernel,np.where(kernel==0))
            background[m][n] = np.mean(kernel1)
background = cv2.medianBlur(background, 3)
plt.imshow(background,cmap='gray')
plt.imshow(b_c1_,cmap='gray')

add_img = np.zeros((28,165))
add_img[np.where(b_c1_==0)]=c1[np.where(b_c1_==0)]
add_img[np.where(add_img==0)]=background[np.where(add_img==0)]-20
plt.imshow(add_img,cmap='gray')
