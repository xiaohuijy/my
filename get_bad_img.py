import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
img = cv2.imread('2_crop.bmp')
def get_back_ground(img):
    c1 = img
    y,x = c1.shape
    b_c1_ = cv2.adaptiveThreshold(c1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 10)
    b_c1 = cv2.erode(b_c1_,kernel = np.ones((3,3)))
    background = c1.copy()
    background[np.where(b_c1==0)]=0
    for m in range(y):
        for n in range(x):
            if background[m][n]==0:
                y0 = (m-3 if m-3>=0 else 0)
                y1 = (m+4 if m+4<=28 else y)
                x0 = (n-3 if n-3>=0 else 0)
                x1 = (n+4 if n+4<=165 else x)
                kernel = background[y0:y1,x0:x1].flatten()
                kernel1 = np.delete(kernel,np.where(kernel==0))
                background[m][n] = np.mean(kernel1)
    background = cv2.medianBlur(background, 3)
    return background
def get_bad_img(img):
    random_num = random.randint(30,50)
    m,n = img.shape
    temp = np.zeros((m,n),dtype='uint8')
    temp[:random_num,:] = img[:random_num,:]
    background = img[random_num:,:]
    background = get_back_ground(background)
    temp[random_num:,:] = background
    return temp
bad = get_bad_img(img[:,:,0])
plt.imshow(bad,cmap='gray')
