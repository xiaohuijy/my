import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import time
def binary(crop):
    m,n = crop.shape
    data_ = []
    for i_ in range(m):
        for j_ in range(n):
            x = crop[i_,j_]
            data_.append([x/256])
    crop = np.mat(data_)    
    label = KMeans(n_clusters=2).fit_predict(crop)  #图片聚成2类
    color_list = [0,255]
    label = label.reshape([m,n])
    w_new = np.zeros((m,n))
    for i_1 in range(m):                          #根据所属类别给图片添加灰度
        for j_1 in range(n):
            w_new[i_1,j_1] = color_list[label[i_1][j_1]]
    if sum(w_new.flatten()==0)>sum(w_new.flatten()==255):
        w_new = 255-w_new
    return w_new
    
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

img = cv2.imread('./img/3.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#crop = img[148:210,262:445]
#crop = img[148:210,266:447]
crop = img[147:207,261:441]
#crop = img[148:210,262:445]
#crop = img[150:207,250:433]
plt.imshow(crop,cmap='gray')

st = time.time()
s1 = binary(crop)
print(time.time()-st)
plt.imshow(s1,cmap='gray')

st = time.time()
s2 = unevenLightCompensate(crop,blockSize=32)
s2 = binary(s2)
print(time.time()-st)
plt.imshow(s2,cmap='gray')

st = time.time()
ret, s3 = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
print(time.time()-st)
plt.imshow(s3,cmap='gray')

st = time.time()
s4 = unevenLightCompensate(crop,blockSize=32)
ret, s4 = cv2.threshold(s4, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
print(time.time()-st)
plt.imshow(s4,cmap='gray')

st = time.time()
s5 =  cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 10)
print(time.time()-st)
plt.imshow(s5,cmap='gray')
