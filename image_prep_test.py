import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#some segmentation experiments
local_path = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(local_path+'/data/camera_examples/M-999i/15100_SpruceSwampCreek_051018_062018 (1).JPG')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(img,25,50,apertureSize=3)
lines = cv2.HoughLines(edges,1,np.pi/180,200)
img = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + img.shape[0]*(-b))
    y1 = int(y0 + img.shape[1]*(a))
    x2 = int(x0 - img.shape[0]*(-b))
    y2 = int(y0 - img.shape[1]*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),40)
    # cv2.circle(img,(x1,y1),100,(0,0,255))
    print((x1,y1,x2,y2))
cv2.circle(img,(img.shape[0]/2,img.shape[1]/2),100,(0,0,255),20)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()