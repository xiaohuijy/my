import cv2
import numpy as np
img = cv2.imread('2.jpg')
c = img[2:30,4:25]
c = cv2.resize(c,(21,28))
cv2.imwrite('44.jpg',c)
