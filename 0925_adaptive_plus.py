import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.cluster import KMeans
path_file = glob('produce/*.*')

for each_file in path_file:
    img0 = cv2.imread(each_file)
    img0 = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)    
    #img = cv2.adaptiveThreshold(img0,1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 25)
    mid = img0.shape[0]//2
    


    crop_each_file = each_file.replace('produce','produce_crop')
    
    img_c1 = img0[:mid]
    img_c2 = img0[mid:] 
    if img0.shape[0]%2==1:
        img_c1 = img0[:mid]
        img_c2 = img0[mid:-1] 
    
    c1 = cv2.adaptiveThreshold(img_c1,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 15)     
    c2 = cv2.adaptiveThreshold(img_c2,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 15)
    
    compare_c1 = cv2.adaptiveThreshold(img_c1,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 25)     
    compare_c2 = cv2.adaptiveThreshold(img_c2,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 25)


    def adaptive_plus(src,binary_src,kernel_size = 25,cc=30):
        img_temp = src.copy()
        binary_temp = binary_src.copy()
        kernel_size = kernel_size//2
        ss = np.ones((img_temp.shape),dtype = 'uint8')*255
        for m in range(img_temp.shape[0]):
            for n in range(img_temp.shape[1]):
                if binary_temp[m][n]==0:
                    ymin,ymax,xmin,xmax = m-kernel_size,m+kernel_size+1,n-kernel_size,n+kernel_size+1
                    ymin = (ymin if ymin>=0 else 0)
                    ymax = (ymax if ymax<=img_temp.shape[0] else img_temp.shape[0])
                    xmin = (xmin if xmin>=0 else 0)
                    xmax = (xmax if xmax<=img_temp.shape[1] else img_temp.shape[1])
                    crop_img = img_temp[ymin:ymax,xmin:xmax]
                    crop_binary = binary_temp[ymin:ymax,xmin:xmax]
                    crop_mean = np.array(sorted(crop_img[np.where(crop_binary==0)])[5:30]).mean()
# =============================================================================
#                     temp_ = []
#                     ssss = crop_img[np.where(crop_binary==0)].tolist()
#                     for i in ssss:
#                         temp_.append([i])
#                     clf = KMeans(n_clusters=2,max_iter=10)
#                     label = clf.fit(temp_)
#                     crop_mean = np.min(clf.cluster_centers_)
# =============================================================================
                    
                    
                    if abs(img_temp[m][n] - crop_mean)<cc:
                        ss[m][n]=0
        ss = cv2.erode(ss,kernel=np.ones((3,1)),anchor=(0,1)) 
        #ss = cv2.erode(ss,kernel=np.ones((2,2))) 
        plt.imshow(ss,cmap='gray')
        return ss
                
    c1_final = adaptive_plus(img_c1,c1)
    c2_final = adaptive_plus(img_c2,c2)
# =============================================================================
#     cv2.imwrite(crop_each_file[:-4]+'_0.jpg',c1_final)
#     cv2.imwrite(crop_each_file[:-4]+'_1.jpg',c2_final)
# =============================================================================
    output = np.concatenate((c1_final,c2_final),axis=1)
    compare_output = np.concatenate((compare_c1,compare_c2),axis=1)
    output = np.concatenate((output,compare_output),axis=0)
    cv2.imwrite(crop_each_file,output)
    
