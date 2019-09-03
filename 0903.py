import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('img/1.bmp')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
c1 = img[149:177,261:426]
c2 = img[178:206,261:444]
plt.imshow(c2)

def split_crop(img_temp,axis_crop):
    img = img_temp.copy()
    img[np.where(img==0)]=1
    img[np.where(img==255)]=0
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

ss =cv2.adaptiveThreshold(c2,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 15)
ss = cv2.erode(ss,np.ones((3,1)),iterations =1,anchor=(0,1))
ss = cv2.dilate(ss,np.ones((1,2)),iterations =1,anchor=(1,0))
#ss = cv2.dilate(ss,np.array([[1],[1],[1]]))
#ss[np.where(ss==0)]=1
#ss[np.where(ss==255)]=0
drc_g = split_crop(ss,0)
crop_pic = []
ymin,ymax = 0,ss.shape[0]
for list_each in drc_g:
    crop_pic.append(c2[ymin:ymax,list_each[0]-1:list_each[1]+1])

plt.imshow(ss,cmap='gray')
