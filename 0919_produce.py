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
for i in range(63):
    path_file.append('all/{0}.bmp'.format(i))


match_to_output = ['0','1','2','3','4','5','6','7','8','9',
                   'a','b','c','d','e','f','g','h','i','j','k','l','m','n',
                   'o','p','q','r','s','t','u','v','w','x','y','z',
                   'A','B','C','D','E','F','G','H','I','J','K','L','M','N',
                   'O','P','Q','R','S','T','U','V','W','X','Y','Z',':']

dict_match = dict(enumerate(match_to_output))
def produce(width,height):
    choose_img = []
    choose_img_num = []
    y_up = random.randint(1,3)
    y_down = 4-y_up
    for i in range(8):
        index = random.randint(0,62)
        img = cv2.imread(path_file[index])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        shape_y,shape_x = img.shape
        #img = cv2.erode(img,np.ones((2,2)))
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
        img_each = np.concatenate((np.ones((y_up,shape_x))*255,img,np.ones((y_down,shape_x))*255),axis=0)
        choose_img.append(img_each)
    
    x_left = random.randint(1,2)
    x_right = random.randint(1,2)
    create_img = np.ones((32,x_left))*255
    is_add_space = random.randint(0,3)
    for i in range(5):
        r = random.randint(1,3)
        create_img = np.concatenate((create_img,choose_img[i],np.ones((32,r))*255),axis=1)
    if is_add_space==0:
        create_img = np.concatenate((create_img,np.ones((32,15))*255),axis=1)
    for i in range(3):
        r = random.randint(1,3)
        create_img = np.concatenate((create_img,choose_img[i+5],np.ones((32,r))*255),axis=1)
    
    create_img = np.concatenate((create_img,np.ones((32,x_right))*255),axis=1)
    
    #plt.imshow(create_img,cmap='gray')
    prob = random.uniform(0.005,0.03)
    sss = sp_noise(create_img,prob)
    content = dict_match[choose_img_num[0]]+dict_match[choose_img_num[1]]+dict_match[choose_img_num[2]]+dict_match[choose_img_num[3]]+dict_match[choose_img_num[4]]+dict_match[choose_img_num[5]]+dict_match[choose_img_num[6]]+dict_match[choose_img_num[7]]
    
    sss = cv2.resize(sss,(width,height),interpolation=cv2.INTER_NEAREST)
    temp_sss = np.zeros((height,width,1), dtype=np.uint8)
    temp_sss[:,:,0]=sss
    #sss = cv2.cvtColor(sss,cv2.COLOR_GRAY2BGR)
    #sss = np.reshape(sss,(height,width,1))
    return temp_sss,content                        

# =============================================================================
# img,t = produce(256,64) 
# plt.imshow(img[:,:,0],cmap='gray')
# plt.title(t)
# =============================================================================
