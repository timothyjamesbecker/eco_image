import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import mysql_connector as msc

#uses the
def get_seg_line(img,low=50,high=150,average=True,offset=0.8,vertical=False):
    seg = [0,0,0,0]
    edges = cv2.Canny(img,low,high,apertureSize=3)
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    for rho,theta in lines[0]:
        a,b   = np.cos(theta),np.sin(theta)
        x0,y0 = a*rho,b*rho
        seg[0] = min(img.shape[1],max(0,int(round(x0+img.shape[1]*(-b)))))
        seg[1] = min(img.shape[1],max(0,int(round(y0+img.shape[0]*(a)))))
        seg[2] = min(img.shape[1],max(0,int(round(x0-img.shape[1]*(-b)))))
        seg[3] = min(img.shape[1],max(0,int(round(y0-img.shape[0]*(a)))))
    if average and vertical:
        seg[0] = int(round(np.mean([seg[0],seg[2]])))
        seg[0] -= int(round((img.shape[0]-seg[1])*offset))
        seg[2] = seg[0]
    if average and not vertical:
        seg[1] = int(round(np.mean([seg[1],seg[3]])))
        seg[1] -= int(round((img.shape[0]-seg[1])*offset))
        seg[3] = seg[1]
    return seg

#seg is one of four orientations t- r| b- |l img needs a cut first
def crop_seg(img,seg,final=(640,480)):
    if np.abs(seg[0]-seg[2]) > np.abs(seg[1]-seg[3]):    #horizontal
        if np.abs(img.shape[0]-seg[1]) < seg[1]: #bottom orientation
            img = img[0:seg[1]+1,:,:]
        else:                                    #top    orientation
            img = img[seg[1]:,:,:]
    else:                                                 # vertical
        if np.abs(img.shape[1]-seg[2])<seg[2]:   #right  orientation
            img = img[:,0:seg[2]+1,:]
        else:                                    #left   orientation
            img = img[:,seg[2]:,:]
    return img

def resize(img,width=640,height=480,interp=cv2.INTER_CUBIC):
    h_scale = height/(img.shape[0]*1.0)
    w_scale =  width/(img.shape[1]*1.0)
    if w_scale<h_scale:
        dim = (int(round(img.shape[1]*h_scale)),int(round(img.shape[0]*h_scale)))
        img = cv2.resize(img,dim)
        d = int(round((img.shape[1]-width)/2.0))
        img = img[:,d:(img.shape[1]-d),:]
    else:
        dim = (int(round(img.shape[1]*w_scale)),
               int(round(img.shape[0]*w_scale)))
        img = cv2.resize(img,dim)
        d = int(round((img.shape[0]-height)))
        img = img[d:img.shape[0],:,:]
    return img

def plot(img):
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.show()

#some segmentation experiments
camera_type = 'M-999i'
local_path = os.path.dirname(os.path.abspath(__file__))
camera_folders = glob.glob(local_path+'/data/camera_examples/*')
cameras = {c.rsplit('/')[-1]:c for c in camera_folders}
img = cv2.imread(glob.glob(cameras[camera_type]+'/*')[0])
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
if camera_type.startswith('M-'):
    offset = 0.10
else:
    if camera_type.startswith('BUSH'):
        offset = 0.9
    else:
        offset = 0.75
# plot(img)
seg = get_seg_line(img,offset=offset)
img = crop_seg(img,seg)
# plot(img)
web = resize(img,width=1280,height=720)
# plot(web)
thb = resize(img,width=256,height=192)
plot(thb)
#some db experiements
