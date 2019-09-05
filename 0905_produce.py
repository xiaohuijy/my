import cv2
import numpy as np
import random
from glob import glob
import matplotlib.pyplot as plt

def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

path_file = []
for i in range(9):
    path_file.append('all/{0}.jpg'.format(i))



for set_num in range(10):
    choose_img = []
    choose_img_num = []
    y_up = random.randint(1,9)
    y_down = 10-y_up
    for i in range(8):
        index = random.randint(0,8)
        img = cv2.imread(path_file[index])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        
        img[np.where(img>100)]=255
        img[np.where(img<100)]=0
        
        is_trans = random.randint(0,12)

        if is_trans==0:
            img = cv2.dilate(img,np.ones((1,2)),anchor=(1,0))
        elif is_trans==1:
            img = cv2.dilate(img,np.ones((2,1)),anchor=(0,1))
        elif is_trans==2:
            img = cv2.dilate(img,np.ones((2,2)))
        elif is_trans==3:
            img = cv2.erode(img,np.ones((1,2)),anchor=(1,0))
        elif is_trans==4:
            img = cv2.erode(img,np.ones((2,1)),anchor=(0,1))
        elif is_trans==5:
            img = cv2.erode(img,np.ones((2,2)))
        choose_img_num.append(index)
        img_each = np.concatenate((np.ones((y_up,21))*255,img,np.ones((y_down,21))*255),axis=0)
        choose_img.append(img_each)
    
    x_left = random.randint(2,5)
    x_right = random.randint(2,5)
    create_img = np.ones((38,x_left))*255
    for i in range(8):
        r = random.randint(0,2)
        create_img = np.concatenate((create_img,choose_img[i],np.ones((38,r))*255),axis=1)
    create_img = np.concatenate((create_img,np.ones((38,x_right))*255),axis=1)
    
    #plt.imshow(create_img,cmap='gray')
    prob = random.uniform(0.005,0.02)
    sss = sp_noise(create_img,0.005)
    #plt.imshow(sss,cmap='gray')
    save_name = 'created/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}.jpg'.format(choose_img_num[0],
                         choose_img_num[1],
                         choose_img_num[2],
                         choose_img_num[3],
                         choose_img_num[4],
                         choose_img_num[5],
                         choose_img_num[6],
                         choose_img_num[7])
    cv2.imwrite(save_name,sss)

