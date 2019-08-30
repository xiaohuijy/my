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


import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
img = cv2.imread('./img/good.bmp')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
st = time.time()

ss = img[:,:].copy()
ss[np.where(ss==0)]=1
ss[np.where(ss==255)]=0
def split_crop(img,axis_crop):
    drc = np.array(np.where(img.sum(axis=axis_crop)!=0)).tolist()[0]
    drc_g = []
    k=0
    for i in range(len(drc)-1):
        if drc[i+1]-drc[i]>2:
            drc_g.append(drc[k:i+1])
            k = i+1
    drc_g.append(drc[k:len(drc)]) 
    for i in range(len(drc_g)):
        tmp = []
        tmp.append(drc_g[i][0])
        #tmp.append(0)
        tmp.append(drc_g[i][-1]+1)
        #tmp.append(180)
        drc_g[i] = tmp
    return drc_g
drc_g = split_crop(ss,1)
ss_crop_0 = ss[drc_g[0][0]:drc_g[0][1],:]
ss_crop_1 = ss[drc_g[1][0]:drc_g[1][1],:]

drc_g_0 = split_crop(ss_crop_0,0)
drc_g_1 = split_crop(ss_crop_1,0)
print(time.time()-st)

cc = []
for i in range(len(drc_g_0)):
    cc.append(cv2.resize(ss[drc_g[0][0]:drc_g[0][1],drc_g_0[i][0]:drc_g_0[i][1]],(18,27)))
# =============================================================================
# print(np.sum(c0==1),np.sum(c1==1),np.sum(c2==1),np.sum(c3==1),
# np.sum(c4==1),np.sum(c5==1),np.sum(c6==1),np.sum(c7==1))
# =============================================================================
cc_count = []
cc_count_ = []
for i in range(len(cc)):
    temp = []
    for j in range(3):
        for k in range(3):
            temp.append(int(np.sum(cc[i][9*j:9*j+9,6*k:6*k+6])))
    cc_count_.append(temp)
    cc_count.append([temp[1]-temp[0],temp[2]-temp[1],temp[4]-temp[3],temp[5]-temp[4],temp[7]-temp[6],temp[8]-temp[7],
                 temp[3]-temp[0],temp[6]-temp[3],temp[4]-temp[1],temp[7]-temp[4],temp[5]-temp[2],temp[8]-temp[5]])


print(time.time()-st)

def count_mean_square(list1,list2):
    count = 0
    for i,j in zip(list1,list2):
        count+= (i-j)**2
    return count
