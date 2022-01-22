import os
import sys
import exifread
import piexif
import cv2
import glob
import math
import datetime as dt
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

#utils------------------------------------------------------------------------
import utils


def local_path():
    return '/'.join(os.path.abspath(__file__).split('/')[:-1])+'/'

def read_exif_tags(path,tag_set='all'):
    tags,T = {},{}
    with open(path,'rb') as f: tags = exifread.process_file(f)
    if tag_set=='all': tag_set = set(list(tags.keys()))
    for t in sorted(list(tags.keys())):
        if t in tag_set and type(tags[t]) is not str:
            if type(tags[t]) is not bytes:
                tag_value = tags[t].values
                if type(tag_value) is list: tag_value = ','.join([str(v) for v in tag_value])
                if type(tag_value) is str: tag_value = tag_value.rstrip(' ')
                T[t] = str(tag_value)
    return T

def set_exif_tags(path,tag_set):
    return True

def read_crop_resize(img_path,width=600,height=200):
    img       = cv2.imread(img_path)
    seg_line  = [0,int(img.shape[0]*0.925),img.shape[1],int(img.shape[0]*0.925)]
    clip_img  = crop_seg(img,seg_line)
    new_img   = resize(clip_img,width=width,height=height)
    return new_img

def get_camera_seg_mult(camera_type):
    if camera_type.upper().startswith('MOULTRIE'):
        offset = 0.10
    elif camera_type.upper().startswith('BUSHNELL'):
        offset = 0.925
    elif camera_type.upper().startswith('SPYPOINT'):
        offset = 0.75
    elif camera_type.upper().startswith('GOPRO'):
        offset = 0.75
    else:
        offset = 0.5
    return offset

def get_seg_line(img,low=50,high=100,average=True,mult=0.8,vertical=False):
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
        seg[0] -= int(round((img.shape[0]-seg[1])*mult))
        seg[2] = seg[0]
    if average and not vertical:
        seg[1] = int(round(np.mean([seg[1],seg[3]])))
        seg[1] -= int(round((img.shape[0]-seg[1])*mult))
        seg[3] = seg[1]
    return seg

def get_rotation_pad(img,luma_thresh=10):
    imgL = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
    lpad,rpad  = 0,0
    for j in range(imgL.shape[1]//2):
        i = 0
        while i<imgL.shape[0]//4 and imgL[i,j]<luma_thresh: i+=1
        if i>lpad: lpad=i
    if lpad==i:lpad=0
    for j in range(imgL.shape[1]//2,imgL.shape[1],1):
        i = 0
        while i<imgL.shape[0]//4 and imgL[i,j]<luma_thresh: i+=1
        if i>rpad: rpad=i
    if rpad==i:rpad=0
    return max(lpad,rpad)
#utils------------------------------------------------------------------------

#tests:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def chroma_dropped(img,cutoff=5):
    dropped = False
    cvt = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    cr_std = np.std(cvt[:,:,1])
    cb_std = np.std(cvt[:,:,2])
    if cr_std<cutoff and cb_std<cutoff: dropped = True
    return dropped

def too_dark(img,cutoff=40):  #night image was too dark
    dark = False
    luma = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
    w,h = luma.shape[0],luma.shape[1]
    z1,z2,z3,z4 = np.mean(luma[:w//2,:h//2]),np.mean(luma[:w//2,h//2:]),np.mean(luma[w//2:,:h//2]),np.mean(luma[w//2:,h//2:])
    zs = [(1 if z1<cutoff else 0),(1 if z2<cutoff else 0),(1 if z3<cutoff else 0),(1 if z4<cutoff else 0)]
    if sum(zs)>3: dark = True
    return dark

def too_light(img,cutoff=180): #overexposure
    light = False
    luma = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
    w,h = luma.shape[0],luma.shape[1]
    z1,z2,z3,z4 = np.mean(luma[:w//2,:h//2]),np.mean(luma[:w//2,h//2:]),np.mean(luma[w//2:,:h//2]),np.mean(luma[w//2:,h//2:])
    zs = [(1 if z1>cutoff else 0),(1 if z2>cutoff else 0),(1 if z3>cutoff else 0),(1 if z4>cutoff else 0)]
    if sum(zs)>2: light = True
    return light

def blurred(img,cutoff=75.0,ksize=3):    #too much image blur
    blur = False
    luma = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
    w,h = luma.shape[0],luma.shape[1]
    d1 = cv2.Laplacian(luma[:w//2,:h//2],ddepth=cv2.CV_64F,ksize=ksize)
    d2 = cv2.Laplacian(luma[:w//2,h//2:],ddepth=cv2.CV_64F,ksize=ksize)
    d3 = cv2.Laplacian(luma[w//2:,:h//2],ddepth=cv2.CV_64F,ksize=ksize)
    d4 = cv2.Laplacian(luma[w//2:,h//2:],ddepth=cv2.CV_64F,ksize=ksize)
    ds = [(1 if np.std(d1)<cutoff else 0),(1 if np.std(d2)<cutoff else 0),(1 if np.std(d3)<cutoff else 0),(1 if np.std(d4)<cutoff else 0)]
    if sum(ds)>2: blur = True
    return blur

def lens_flare(img,pixel_size=None,verbose=False):
    flare = False
    if pixel_size is None:
        pixel_size = int(round(min(img.shape[0:2])/10))
    area  = int(round(0.25*np.pi*pixel_size**2))
    luma  = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
    blur  = cv2.medianBlur(luma,5)
    sharp = sharpen(blur,amount=2.0)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = area
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.8
    detector = cv2.SimpleBlobDetector_create(params)
    for t in [200,195,190,185,180,175,170,165,160]:
        ret, thresh = cv2.threshold(sharp,t,255,cv2.THRESH_BINARY_INV)
        kpts = detector.detect(thresh)
        if len(kpts)>0: break
    if len(kpts)>0: flare = True
    if verbose:
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(img,kpts,np.zeros((1,1)),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        utils.plot(blobs)
    return flare

#make more robust with hue rotation ???
def luma_rotated(ref,img,sd=3.0,d_min=2.0,d_max=10.0,steps=25,pad=0.2,verbose=False):
    theta,rot = 0.0,False
    angs = np.arange(d_min,d_max,(d_max-d_min)/steps)
    angs = sorted(sorted(angs)+sorted(-1*angs))[::-1]
    refR,imgA = crop(np.copy(ref),pad),crop(np.copy(img),pad)
    ref_diff = np.std(luma_diff(refR,imgA))
    for ang in angs:
        imgR = crop(rotate(np.copy(img),ang),pad)
        new_diff = np.std(luma_diff(refR,imgR))
        if new_diff+sd<ref_diff: theta = ang
    if (theta<0.0 or theta>0.0): rot = True
    return rot
#tests:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#image processing routines]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
def crop(img,pad=0.2,pixels=None):
    if pixels is None:
        h_pad,w_pad  = int(round(pad*img.shape[0]//2)),int(round(pad*img.shape[1]//2))
        img1 = img[h_pad:(img.shape[0]-h_pad+1),w_pad:(img.shape[1]-w_pad+1),:]
    else:
        img1 = img[pixels:(img.shape[0]-pixels+1),pixels:(img.shape[1]-pixels+1),:]
    return img1

def crop_seg(img,seg):
    if np.abs(seg[0]-seg[2]) > np.abs(seg[1]-seg[3]):    #horizontal
        if np.abs(img.shape[0]-seg[1]) < seg[1]: #bottom orientation
            img1 = img[0:seg[1]+1,:,:]
        else:                                    #top    orientation
            img1 = img[seg[1]:,:,:]
    else:                                                 # vertical
        if np.abs(img.shape[1]-seg[2])<seg[2]:   #right  orientation
            img1 = img[:,0:seg[2]+1,:]
        else:                                    #left   orientation
            img1 = img[:,seg[2]:,:]
    return img1

def resize(img,width=640,height=480,interp=cv2.INTER_CUBIC):
    h_scale = height/(img.shape[0]*1.0)
    w_scale =  width/(img.shape[1]*1.0)
    if w_scale<h_scale:
        dim = (int(round(img.shape[1]*h_scale)),int(round(img.shape[0]*h_scale)))
        img1 = cv2.resize(img,dim,interpolation=interp)
        d = int(round((img1.shape[1]-width)/2.0))
        img1 = img1[:,d:(img1.shape[1]-d),:]
    else:
        dim = (int(round(img.shape[1]*w_scale)),
               int(round(img.shape[0]*w_scale)))
        img1 = cv2.resize(img,dim,interpolation=interp)
        d = int(round((img1.shape[0]-height)))
        img1 = img1[d:img1.shape[0],:,:]
    img1 = cv2.resize(img1,(width,height),interpolation=interp)
    return img1

def rotate(image,angle):
  image_center = tuple(np.array(image.shape[1::-1])/2)
  rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
  result = cv2.warpAffine(image,rot_mat,image.shape[1::-1],flags=cv2.INTER_CUBIC)
  return result

def multi_tilt_resize(img,width=640,height=480,n=5,deg=45,translate=False,interp=cv2.INTER_CUBIC):
    imgs = []
    degs = [x for x in range(-1*deg,deg,(deg*2)//(n-1))]
    if len(degs)<n: degs += [deg]
    for j in range(n):
        M = cv2.getRotationMatrix2D(((img.shape[0]-1)/2.0,(img.shape[0]-1)/2.0),degs[j],1)
        new_img = cv2.warpAffine(img,M,img.shape[0:2][::-1])
        a,b = int(round(new_img.shape[0]/2)),int(round(new_img.shape[1]/2))
        if abs(degs[j])<20:
            sel_img = new_img[a//2:3*a//2,b//2:3*b//2,:]
        else:
            sel_img = new_img[a//3:a,b//3:b,:]
        h_scale = height/(sel_img.shape[0]*1.0)
        w_scale = width/(sel_img.shape[1]*1.0)
        if w_scale<h_scale:
            if translate:
                dim = (int(round(sel_img.shape[1]*h_scale)),
                       int(round(sel_img.shape[0]*h_scale)))
                img1 = cv2.resize(sel_img,dim,interpolation=interp)
                d = int(round((img1.shape[1]-width)/2.0))
                for i in range(n):
                    x = d//(n-i)
                    y = (img1.shape[1]-2*d)
                    imgs += [cv2.resize(img1[:,x:(x+y),:],(width,height),interpolation=interp)]
            else:
                dim = (int(round(img.shape[1]*h_scale)),
                       int(round(img.shape[0]*h_scale)))
                img1 = cv2.resize(sel_img,dim,interpolation=interp)
                d = int(round((img1.shape[1]-width)/2.0))
                imgs += [cv2.resize([img1[:,d:(img1.shape[1]-d),:]],(width,height),interpolation=interp)]
        else:
            if translate:
                dim = (int(round(sel_img.shape[1]*w_scale)),
                       int(round(sel_img.shape[0]*w_scale)))
                img1 = cv2.resize(sel_img, dim, interpolation=interp)
                d = int(round((img1.shape[0]-height)))
                for i in range(n):
                    x = d//(n-i)
                    y = img1.shape[0]-d
                    imgs += [cv2.resize(img1[x:x+y,:,:],(width,height),interpolation=interp)]
            else:
                dim = (int(round(sel_img.shape[1]*w_scale)),
                       int(round(sel_img.shape[0]*w_scale)))
                img1 = cv2.resize(sel_img,dim,interpolation=interp)
                d = int(round((img1.shape[0]-height)))
                imgs += [cv2.resize(img1[d:img1.shape[0],:,:],(width,height),interpolation=interp)]
    return imgs

def hue_mean(imgs):
    hue_sum = np.zeros((imgs[0].shape[0],imgs[0].shape[1]),dtype=float)
    for img in imgs:
        hue_sum += cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,0]
    hue_sum /= len(imgs)*1.0
    return np.asarray(np.round(hue_sum),dtype=np.uint8)

def sat_mean(imgs):
    sat_sum = np.zeros((imgs[0].shape[0],imgs[0].shape[1]),dtype=float)
    for img in imgs:
        sat_sum += cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,1]
    sat_sum /= len(imgs)*1.0
    return np.asarray(np.round(sat_sum),dtype=np.uint8)

def val_mean(imgs):
    val_sum = np.zeros((imgs[0].shape[0],imgs[0].shape[1]),dtype=float)
    for img in imgs:
        val_sum += cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,2]
    val_sum /= len(imgs)*1.0
    return np.asarray(np.round(val_sum),dtype=np.uint8)

def luma_mean(imgs,equalize=False):
    ycrcb_sum = np.zeros((imgs[0].shape[0],imgs[0].shape[1]),dtype=float)
    if equalize:
        for img in imgs:
            ycrcb_sum += cv2.equalizeHist(cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0])
    else:
        for img in imgs:
            ycrcb_sum += cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
    ycrcb_sum /= len(imgs)*1.0
    return np.asarray(np.round(ycrcb_sum),dtype=np.uint8)

def cr_mean(imgs,equalize=False):
    cr_sum = np.zeros((imgs[0].shape[0],imgs[0].shape[1]),dtype=float)
    if equalize:
        for img in imgs:
            cr_sum += cv2.equalizeHist(cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,1])
    else:
        for img in imgs:
            cr_sum += cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,1]
    cr_sum /= len(imgs)*1.0
    return np.asarray(np.round(cr_sum),dtype=np.uint8)

def cb_mean(imgs,equalize=False):
    cb_sum = np.zeros((imgs[0].shape[0],imgs[0].shape[1]),dtype=float)
    if equalize:
        for img in imgs:
            cb_sum += cv2.equalizeHist(cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,2])
    else:
        for img in imgs:
            cb_sum += cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,2]
    cb_sum /= len(imgs)*1.0
    return np.asarray(np.round(cb_sum),dtype=np.uint8)

def color_enhance(img,hmean,smean,amount=1.0):
    cvt = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    diff       = np.asarray(hmean,dtype=int)-np.asarray(cvt[:,:,0],dtype=int)
    corr       = np.asarray(np.round(diff*amount),dtype=int)
    cvt[:,:,0] = np.asarray(np.asarray(cvt[:,:,0],dtype=int)+corr,dtype=np.uint8)
    diff       = np.asarray(smean,dtype=int)-np.asarray(cvt[:,:,1],dtype=int)
    corr       = np.asarray(np.round(diff*amount),dtype=int)
    cvt[:,:,1] = np.asarray(np.asarray(cvt[:,:,1],dtype=int)+corr,dtype=np.uint8)
    return cv2.cvtColor(cvt,cv2.COLOR_HSV2BGR)

def luma_enhance(img,lmean,amount=1.0,winsize=9,advanced=False):
    cvt          = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    diff         = np.asarray(lmean,dtype=int)-np.asarray(cvt[:,:,0],dtype=int)
    if advanced:
        pos              = np.zeros(diff.shape,dtype=np.uint8)
        pos[diff>0]      = diff[diff>0]
        neg              = np.zeros(diff.shape,dtype=np.uint8)
        neg[diff<0]      = np.abs(diff[diff<0])
        lmean_pad        = np.zeros_like(img)
        lmean_pad[:,:,0] = lmean
        lmean_pad = cv2.cvtColor(lmean_pad,cv2.COLOR_YCrCb2BGR)
        flow = luma_dense_optical_flow(lmean_pad,img,winsize=winsize)[:,:,2]
        pos_corr = np.asarray(np.round(pos*amount*np.invert(flow)/255.0),dtype=np.uint8)
        neg_corr = -1*np.asarray(np.round(pos*amount*np.invert(flow)/255.0),dtype=int)
        cvt[:,:,0] = np.asarray(np.asarray(cvt[:,:,0],dtype=int)+pos_corr+neg_corr,dtype=np.uint8)
    else:
        corr = np.asarray(np.round(diff*amount),dtype=int)
        cvt[:,:,0] = np.asarray(np.asarray(cvt[:,:,0],dtype=int)+corr,dtype=np.uint8)
    cvt = cv2.cvtColor(cvt, cv2.COLOR_YCrCb2BGR)
    return cvt

def chroma_enhance(img,crmean,cbmean,amount=1.0):
    cvt = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    cr_diff = np.asarray(crmean,dtype=int)-np.asarray(cvt[:,:,1],dtype=int)
    cr_corr = np.asarray(np.round(cr_diff*amount),dtype=int)
    cb_diff = np.asarray(cbmean,dtype=int)-np.asarray(cvt[:,:,2],dtype=int)
    cb_corr = np.asarray(np.round(cb_diff*amount),dtype=int)
    cvt[:,:,1] = np.asarray(np.asarray(cvt[:,:,1],dtype=int)+cr_corr,dtype=np.uint8)
    cvt[:,:,2] = np.asarray(np.asarray(cvt[:,:,2],dtype=int)+cb_corr,dtype=np.uint8)
    return cv2.cvtColor(cvt,cv2.COLOR_YCrCb2BGR)

def hue_diff(img1,img2):
    hue1 = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)[:,:,0]
    hue2 = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)[:,:,0]
    diff = np.asarray(hue1,dtype=int)-np.asarray(hue2,dtype=int)
    pos  = np.zeros_like(diff)
    pos[diff>0] = diff[diff>0]
    neg  = np.zeros_like(diff)
    neg[diff<0] = np.abs(diff[diff<0])
    bgr = np.zeros_like(img1)
    bgr[:,:,2] = pos[:]
    bgr[:,:,1] = neg[:]
    return bgr

def hue_dense_optical_flow(img1,img2,winsize=9,equalization=False):
    prev = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)[:,:,0]
    next = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)[:,:,0]
    if equalization:
        prev = cv2.equalizeHist(prev)
        next = cv2.equalizeHist(next)
    flow = cv2.calcOpticalFlowFarneback(prev,next,None,
                                        pyr_scale=0.5,levels=3,winsize=winsize,
                                        iterations=6,poly_n=7,poly_sigma=1.5,
                                        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    mag,ang = cv2.cartToPolar(flow[:,:,0],flow[:,:,1])
    hsv = np.zeros_like(img1)
    hsv[:,:,1] = 255
    hsv[:,:,0] = ang*(180/(np.pi/2))
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return bgr

def luma_diff(img1,img2): #green is positive luma and red is negative luma
    lum1 = cv2.cvtColor(img1,cv2.COLOR_BGR2YCrCb)[:,:,0]
    lum2 = cv2.cvtColor(img2,cv2.COLOR_BGR2YCrCb)[:,:,0]
    diff = np.asarray(lum1,dtype=int)-np.asarray(lum2,dtype=int)
    pos  = np.zeros_like(diff)
    pos[diff>0] = diff[diff>0]
    neg  = np.zeros_like(diff)
    neg[diff<0] = np.abs(diff[diff<0])
    bgr = np.zeros_like(img1)
    bgr[:,:,2] = pos[:]
    bgr[:,:,1] = neg[:]
    return bgr

def luma_dense_optical_flow(img1,img2,winsize=9,equalization=True):
    prev = cv2.cvtColor(img1,cv2.COLOR_BGR2YCrCb)[:,:,0]
    next = cv2.cvtColor(img2,cv2.COLOR_BGR2YCrCb)[:,:,0]
    if equalization:
        prev = cv2.equalizeHist(prev)
        next = cv2.equalizeHist(next)
    flow = cv2.calcOpticalFlowFarneback(prev,next,None,
                                        pyr_scale=0.5,levels=5,winsize=winsize,
                                        iterations=6,poly_n=7,poly_sigma=1.5,
                                        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    mag,ang = cv2.cartToPolar(flow[:,:,0],flow[:,:,1])
    hsv = np.zeros_like(img1)
    hsv[:,:,1] = 255
    hsv[:,:,0] = ang*(180/(np.pi/2))
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return bgr

def sharpen(image,kernel_size=(5, 5),sigma=1.0,amount=1.0,threshold=0):
    blurred = cv2.GaussianBlur(image,kernel_size,sigma)
    sharpened = float(amount+1)*image - float(amount)*blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image-blurred) < threshold
        np.copyto(sharpened,image,where=low_contrast_mask)
    return sharpened

def color_equalization(img):
    ycrcb_img = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    img_chn   = cv2.split(ycrcb_img)
    img_chn[0] = cv2.equalizeHist(img_chn[0])
    cv2.merge(img_chn[0],ycrcb_img)
    img1 = cv2.cvtColor(ycrcb_img,cv2.COLOR_YCR_CB2BGR)
    return img1

def homography(ref,img,f_num=10000,percent=0.75,pixel_dist=100.0,luma=True,orb=False,homo=None):
    if homo is None:
        imgA,imgB = np.copy(img),np.copy(ref)
        if luma:
            imgA = cv2.cvtColor(imgA,cv2.COLOR_BGR2YCrCb)[:,:,0]
            imgB = cv2.cvtColor(imgB,cv2.COLOR_BGR2YCrCb)[:,:,0]
        detector   = cv2.ORB_create(f_num)
        kpts1      = detector.detect(imgA,None)
        kpts2      = detector.detect(imgB,None)
        if orb: descriptor = cv2.ORB_create(f_num)
        else:   descriptor = cv2.xfeatures2d.BEBLID_create(percent)
        kpts1,des1 = descriptor.compute(imgA,kpts1)
        kpts2,des2 = descriptor.compute(imgB,kpts2)
        #prefilter points that are too far away?
        matcher    = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches    = sorted(matcher.match(des1,des2,None),key=lambda x: x.distance)
        matches    = matches[:max(1,int(round(percent*len(matches))))]
        imatches   = cv2.drawMatches(imgA,kpts1,imgB,kpts2,matches,None)
        pts1,pts2  = np.zeros((len(matches),2),dtype=np.float32),np.zeros((len(matches),2),dtype=np.float32)
        for i in range(len(matches)):
            pts1[i,:] = kpts1[matches[i].queryIdx].pt
            pts2[i,:] = kpts2[matches[i].trainIdx].pt
        # try to find a homography...............................
        homo,mask = cv2.findHomography(pts1,pts2,cv2.USAC_DEFAULT)
    if homo is not None:
        h_pts = [[0,0],[img.shape[1],0],[img.shape[0],img.shape[1]],[0,img.shape[1]]]
        p_pts,d_pts  = [],[]
        for pts in h_pts:
            col = np.ones((3,1),dtype=np.float64)
            col[0:2,0] = pts
            col = np.dot(homo,col)
            col /= col[2,0]
            p_pts += [[int(round(col[0][0])),int(round(col[1][0]))]]
            d_pts += [np.sqrt((pts[0]-col[0][0])**2+(pts[1]-col[1][0])**2)]
        if np.sum(d_pts)<pixel_dist:
            img3 = cv2.warpPerspective(img,homo,(img.shape[1],img.shape[0]))
        else: #throw away the homography
            img3 = np.copy(img)
            homo = None
    else:
        img3 = np.copy(img)
    return img3,homo

def BGR_to_l1l2l3(img):
    cvt   = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    R,G,B = cv2.split(cvt)
    r_g2  = (np.array(R,dtype=int)-np.array(G,dtype=int))**2
    r_b2  = (np.array(R,dtype=int)-np.array(B,dtype=int))**2
    g_b2  = (np.array(G,dtype=int)-np.array(B,dtype=int))**2
    d     = r_g2+r_b2+g_b2
    cvt[:,:,0] = np.asarray(np.round(255*(r_g2/(d+1e-12))),dtype=np.uint8)
    cvt[:,:,1] = np.asarray(np.round(255*(r_b2/(d+1e-12))),dtype=np.uint8)
    cvt[:,:,2] = np.asarray(np.round(255*(g_b2/(d+1e-12))),dtype=np.uint8)
    return cvt

def BGR_to_c1c2c3(img):
    return True #:::TO DO:::

def BGR_to_bgr(img):
    cvt = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    R,G,B = cv2.split(cvt)
    rgb_sum = np.array(R,dtype=int)+np.array(G,dtype=int)+np.array(B,dtype=int)
    cvt[:,:,0] = np.asarray(np.round(255*B/(rgb_sum+1e-12)),dtype=np.uint8)
    cvt[:,:,1] = np.asarray(np.round(255*G/(rgb_sum+1e-12)),dtype=np.uint8)
    cvt[:,:,2] = np.asarray(np.round(255*R/(rgb_sum+1e-12)),dtype=np.uint8)
    return cvt
#image processing routines---------------------------------------------------------------------------

def plot(img,size=(12,9)):
    plt.rcParams["figure.figsize"] = size
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.show()

def blob2img(blob):
    return cv2.imdecode(np.frombuffer(blob,dtype=np.uint8),-1)

def img2blob(img):
    return cv2.imencode('.jpg',img)[1].tostring()

def imgs2pickle(imgs):
    return True

def load_data_simple(in_dir,gray_scale=True,split=0.15):
    data,labels = [],[]
    if gray_scale:
        for path in sorted(glob.glob(in_dir+'/*/*.jpg')+glob.glob(in_dir+'/*/*.JPG')):
            labels += [int(path.rsplit('label_')[-1].rsplit('/')[0])]
            data   += [cv2.imread(path,cv2.IMREAD_GRAYSCALE)]
        l,(h,w) = int(round(split*len(data))),data[0].shape
        test_idx  = np.asarray(sorted(np.random.choice(range(len(data)),l,replace=False)))
        train_idx = np.asarray(sorted(list(set(range(len(data))).difference(set(list(test_idx))))))
        test_labels  = np.asarray(labels, dtype='uint8')[test_idx]
        train_labels = np.asarray(labels, dtype='uint8')[train_idx]
        test_data    = np.ndarray(shape=(len(test_idx),h,w,1),dtype='uint8')
        train_data   = np.ndarray(shape=(len(train_idx),h,w,1),dtype='uint8')
        for i in range(len(test_data)): test_data[i,:,:,0]  = data[test_idx[i]]
        for i in range(len(train_data)):train_data[i,:,:,0] = data[train_idx[i]]
    else:
        for path in sorted(glob.glob(in_dir+'/*/*.jpg')+glob.glob(in_dir+'/*/*.JPG')):
            labels += [int(path.rsplit('label_')[-1].rsplit('/')[0])]
            data   += [cv2.imread(path)]
        l,(h,w,c) = int(round(split*len(data))),data[0].shape
        test_idx     = np.asarray(sorted(np.random.choice(range(len(data)),l,replace=False)))
        train_idx    = np.asarray(sorted(list(set(range(len(data))).difference(set(list(test_idx))))))
        test_labels  = np.asarray(labels,dtype='uint8')[test_idx]
        train_labels = np.asarray(labels,dtype='uint8')[train_idx]
        test_data    = np.ndarray(shape=(len(test_idx),h,w,c),dtype='uint8')
        train_data   = np.ndarray(shape=(len(train_idx),h,w,c),dtype='uint8')
        for i in range(len(test_data)):
            test_data[i,:,:,:] = data[test_idx[i]]
        for i in range(len(train_data)):
            train_data[i,:,:,:] = data[train_idx[i]]
    return (train_data,train_labels),(test_data,test_labels)

#advanced data loader splits with at least one site that is unseen...
#must have a minimum of two sites per label in order to train/test
def load_data_advanced(in_dir,gray_scale=True,split=0.15,enforce_test_label=True,verbose=True):
    L,C,ls = {},{},set([])
    paths = sorted(glob.glob(in_dir+'/*/*.jpg')+glob.glob(in_dir+'/*/*.JPG'))
    for i in range(len(paths)):
        sid   = int(paths[i].rsplit('/')[-1].rsplit('_')[0])
        label = int(paths[i].rsplit('label_')[-1].rsplit('/')[0])
        if sid in L: L[sid] += [[i,label]]
        else:        L[sid]  = [[i,label]]
        if sid in C: C[sid] += [label]
        else:        C[sid]  = [label]
        ls.add(label)
    trn_data,trn_labels,trn_paths,tst_data,tst_labels,tst_paths = [],[],[],[],[],[]
    sids = list(L.keys())
    if enforce_test_label:
        idxs = []
        for i in C:
            if set(C[i])==ls: idxs += [i]
        if len(idxs)<1:
            print("can not enforce test label selection, accuracy measurements will not be accurate....")
            ts = max(1,round(len(L)*split))
            test_sidx  = sorted(list(np.random.choice(sids,ts)))
        else:
            ts = max(1,round(len(idxs)*split))
            test_sidx = sorted(list(np.random.choice(idxs,ts)))
    else:
        ts = max(1,round(len(L)*split))
        test_sidx = sorted(list(np.random.choice(sids,ts)))

    train_sidx = sorted(list(set(sids).difference(set(test_sidx))))
    if verbose: print('sids %s selected for test cases'%(','.join([str(x) for x in test_sidx])))
    if gray_scale:
        for sid in test_sidx:
            for [i,label] in L[sid]:
                tst_labels  += [label]
                tst_data    += [cv2.imread(paths[i],cv2.IMREAD_GRAYSCALE)]
                tst_paths   += [paths[i]]
        for sid in train_sidx:
            for [i,label] in L[sid]:
                trn_labels += [label]
                trn_data   += [cv2.imread(paths[i],cv2.IMREAD_GRAYSCALE)]
                trn_paths  += [paths[i]]
        (h,w) = trn_data[0].shape
        train_labels = np.asarray(trn_labels,dtype='uint8')
        test_labels  = np.asarray(tst_labels,dtype='uint8')
        train_data   = np.ndarray((len(trn_data),h,w,1),dtype='uint8')
        test_data    = np.ndarray((len(tst_data),h,w,1),dtype='uint8')
        for i in range(len(train_data)): train_data[i,:,:,0] = trn_data[i]
        for i in range(len(test_data)):  test_data[i,:,:,0]  = tst_data[i]
    else:
        for sid in test_sidx:
            for [i,label] in L[sid]:
                tst_labels  += [label]
                tst_data    += [cv2.imread(paths[i])]
                tst_paths   += [paths[i]]
        for sid in train_sidx:
            for [i,label] in L[sid]:
                trn_labels += [label]
                trn_data   += [cv2.imread(paths[i])]
                trn_paths  += [paths[i]]
        (h,w,c) = trn_data[0].shape
        train_labels = np.asarray(trn_labels,dtype='uint8')
        test_labels  = np.asarray(tst_labels,dtype='uint8')
        train_data   = np.ndarray((len(trn_data),h,w,c),dtype='uint8')
        test_data    = np.ndarray((len(tst_data),h,w,c),dtype='uint8')
        for i in range(len(train_data)): train_data[i,:,:,:] = trn_data[i]
        for i in range(len(test_data)):  test_data[i,:,:,:]  = tst_data[i]
    return (train_data,train_labels,trn_paths),(test_data,test_labels,tst_paths)

def n_choose_k(n,k):
    k_fact = np.product([i for i in range(1,k+1,1)])
    n_peel = np.product([i for i in range(n-k+1,n+1,1)])
    return int(n_peel/k_fact)

def sids_spectrum(S):
    spectrum = {}
    for sid in S:
        for i in range(len(S[sid])):
            if S[sid][i] in spectrum: spectrum[S[sid][i]] += 1
            else:                spectrum[S[sid][i]]  = 0
    return spectrum

def label_spect_diff(A,B):
    ks = sorted(list(set(A).union(B)))
    D = {}
    for l in ks:
        if l in A:
            D[l] = A[l]
            if l in B: D[l] = A[l]-B[l]
        else: D[l] = -1*B[l]
    return D

def split_diff(A,B,split=0.25):
    D,S,NS = {},label_spect_diff(A,{}),label_spect_diff(A,B)
    for l in NS:
        D[l] = abs(NS[l]/S[l]-split)
    return sum([D[d] for d in D])

#given sids to spectrum, pick the t sites that provide the best spectrum (closest to real spectrum R)
def best_select(S,R,t,rounds=1000,verbose=False):
    best = [1.0,[]] #best[0] is the difference 1.0 is maximal and best[1] is the set of keys
    for j in range(rounds):
        s  = np.random.choice(sorted(S),t,replace=False)
        ss,sn = {r:0 for r in R},0
        for i in range(len(s)):
            for l in S[s[i]]:
                ss[l] += S[s[i]][l]
                sn += S[s[i]][l]
        ss = {l:ss[l]/sn for l in ss}
        diff = sum([abs(R[l]-ss[l]) for l in R])
        if diff<best[0]: best = [diff,sorted(s)]
    if verbose: print('best split after %s rounds had %s spectrum diff'%(rounds,best[0]))
    return best

def partition_train_test_valid(in_dir,class_idx,split=0.15,sub_sample=None,verbose=True):
    paths = sorted(glob.glob(in_dir+'/*/*.jpg')+glob.glob(in_dir+'/*/*.JPG'))
    S,R = {},{}
    for i in range(len(paths)):
        sid   = int(paths[i].rsplit('/')[-1].rsplit('_')[0])
        label = class_idx[int(paths[i].rsplit('label_')[-1].rsplit('/')[0])]
        if label in R: R[label] += 1
        else:          R[label]  = 1
        if sid in S:
            if label in S[sid]: S[sid][label] += 1
            else:               S[sid][label]  = 1
        else:                   S[sid]  = {label:1}
    R = {r:R[r]/len(paths) for r in R}
    test_sids  = best_select(S,R,int(round(split*len(S))),verbose=verbose)[1]
    D = {}
    for s in S:
        if s not in test_sids: D[s] = S[s]
    valid_sids = best_select(D,R,int(round(0.5*split*len(S))),verbose=verbose)[1]
    train_sids = sorted(set(sorted(S)).difference(set(test_sids).union(set(valid_sids))))
    train,test,valid,valid_test = [],[],[],[]
    for i in range(len(paths)):
        sid   = int(paths[i].rsplit('/')[-1].rsplit('_')[0])
        label = class_idx[int(paths[i].rsplit('label_')[-1].rsplit('/')[0])]
        if sid in train_sids:   train += [paths[i]]
        elif sid in test_sids:  test  += [paths[i]]
        elif sid in valid_sids: valid += [paths[i]]
    if sub_sample is not None: #fraction to sub-sample
        train = np.random.choice(train,min(len(train),int(round(sub_sample*len(train)))),replace=False)
        test = np.random.choice(test,min(len(test),int(round(sub_sample*len(test)))),replace=False)
        valid = np.random.choice(valid,min(len(valid),int(round(sub_sample*len(valid)))),replace=False)
    if verbose:
        #rsplit('/')[-1].split('_')[1]
        V = sorted(list(set([v.split('/')[-1].split('_')[1] for v in valid])))
        print('validation sites selected were:%s'%V)
    return train,valid,test

def partition_data_paths(in_dir,class_idx,split=0.15,seed=None,
                         strict_test_sid=False,balance=None,verbose=True):
    if seed is not None: np.random.seed(seed)
    L,C,LC,ls = {},{},{},set([])
    paths = sorted(glob.glob(in_dir+'/*/*.jpg')+glob.glob(in_dir+'/*/*.JPG'))
    for i in range(len(paths)):
        sid   = int(paths[i].rsplit('/')[-1].rsplit('_')[0])
        label = class_idx[int(paths[i].rsplit('label_')[-1].rsplit('/')[0])]
        if sid in L:       L[sid] += [[i,label]]
        else:              L[sid]  = [[i,label]]
        if sid in C:       C[sid] += [label]
        else:              C[sid]  = [label]
        if label in LC: LC[label] += [sid]
        else:           LC[label]  = [sid]
        ls.add(label)
    for l in LC: LC[l] = sorted(list(set(LC[l])))
    trn_paths,tst_paths = [],[]
    sids = list(L.keys())

    #site-per-label-ratio-sampling-balance-----------------------------------
    counts = {}
    for l in LC: counts[l] = 0
    for sid in C:
        lbs = C[sid]
        for l in lbs: counts[l] += 1
    counts = sorted([[counts[c],c] for c in counts],key=lambda x: x[0])[::-1]
    lmax = [x for x in counts[0]]
    sx = counts[1:]
    if lmax[0]>sum([x[0] for x in sx]): lmax[0] = sum([x[0] for x in sx])
    sy = sorted([lmax]+sx,key=lambda x: x[1])
    sx = sorted([x for x in counts],key=lambda x: x[1])
    if balance is not None:
        if verbose: print('using label balancing')
        counts = []
        for i in range(len(sx)):
            if verbose: print('balanced %s to %s of label %s'%(sx[i][0],int(sx[i][0]*min(1.0,sy[i][0]/sx[i][0]*balance)),sx[i][1]))
            if sx[i][0]>=lmax[0]: counts += [[min(1.0,sy[i][0]/sx[i][0]*balance),sx[i][1]]]
            elif sx[i][0]>0.0:    counts += [[min(1.0,sy[i][0]/sx[i][0]),sx[i][1]]]
            else:                 counts += [[0.0,sx[i][1]]]
    else:                         counts = [[1.0,i[1]] for i in counts]
    counts = {x[1]:x[0] for x in counts}
    #site-per-label-ratio-sampling-balance-----------------------------------

    tst_paths,trn_paths,T = [],[],{}
    if strict_test_sid:  # use the sid label spectrum to sample close to split, this will keep entire sites from training
        kl = sorted([[l,len(LC[l])] for l in LC],key=lambda x: x[1]) #sorted keys by number of sids that have that label
        S = sids_spectrum(C) #totals
        c_si,w = [],100
        for l in kl:
            ts,min_split = max(1,int(round(l[0]*split))),[1.0,[]]
            for i in range(min(w,n_choose_k(l[1],ts))): #bounded search by w
                si = list(np.random.choice(LC[l[0]],ts,replace=False))
                NS = sids_spectrum({sid:C[sid] for sid in set(C).difference(set(c_si+si))})
                d = split_diff(S,NS,split=split)/(1.0*len(S))
                if d<min_split[0]: min_split = [d,si]
            c_si += min_split[1]
            c_si = sorted(list(set(c_si)))
        NS = sids_spectrum({sid:C[sid] for sid in set(C).difference(set(c_si))})
        NS = label_spect_diff(NS,{})
        SN = label_spect_diff(S,NS)

        d = split_diff(S,NS,split=split)/(1.0*len(S))
        if verbose:
            print('using strict site hold out with split=%s +|- %s'%(split,round(d,2)))
            print('original spectrum: %s'%S)
            print('training spectrum: %s'%NS)
            print('testing  spectrum: %s'%SN)
        test_sidx  = c_si
        train_sidx = sorted(list(set(sids).difference(set(test_sidx))))
        for sid in L:
            if sid in test_sidx:
                for [i,label] in L[sid]:
                    if np.random.choice([True,False],1,p=[counts[label],1.0-counts[label]]):
                        tst_paths += [paths[i]]
            else:
                for [i,label] in L[sid]:
                    if np.random.choice([True,False],1,p=[counts[label],1.0-counts[label]]):
                        trn_paths += [paths[i]]
        if verbose:
            print('test sites randomly selected were:%s'%c_si)
    else:
        for l in LC:
            ts = max(1,int(round(len(LC[l])*split)))
            test_sidx  = sorted(list(np.random.choice(LC[l],ts,replace=False)))
            train_sidx = sorted(list(set(sids).difference(set(test_sidx))))
            for sid in test_sidx:
                l_idx = np.random.choice(range(len(L[sid])),int(len(L[sid])*counts[l]),replace=False)
                L[sid] = np.asarray(L[sid])
                for [i,label] in L[sid][l_idx]:
                    if label == l: tst_paths   += [paths[i]]
            for sid in train_sidx:
                l_idx = np.random.choice(range(len(L[sid])),int(len(L[sid])*counts[l]),replace=False)
                L[sid] = np.asarray(L[sid])
                for [i,label] in L[sid][l_idx]:
                    if label == l: trn_paths  += [paths[i]]
            T[l] = test_sidx
        if verbose:
            print('test sites randomly selected were:\n'+
                  '\n'.join(['%s:\t'%l+','.join([str(i) for i in T[l]]) for l in sorted(list(T.keys()))]))
    trn_paths = sorted(list(set(trn_paths)))
    tst_paths = sorted(list(set(tst_paths)))
    return trn_paths,tst_paths

def load_data_generator(paths,class_idx,batch_size=64,gray_scale=True,norm=True,offset=-1):
    while True:
        data,labels,l = [],[],len(paths)
        idx = np.random.choice(range(l),size=min(batch_size,l),replace=False)
        if gray_scale:
            for i in idx:
                lab     = int(paths[i].rsplit('label_')[-1].rsplit('/')[0])
                labels += [class_idx[lab]+offset]
                data   += [cv2.imread(paths[i],cv2.IMREAD_GRAYSCALE)]
            y = np.asarray(labels,dtype='uint8')
            (h,w) = data[0].shape
            x = np.ndarray((batch_size,h,w,1),dtype='uint8')
            for i in range(batch_size): x[i,:,:,0] = data[i]
        else:
            for i in idx:
                lab     = int(paths[i].rsplit('label_')[-1].rsplit('/')[0])
                labels += [class_idx[lab]+offset]
                data   += [cv2.imread(paths[i])]
            y = np.asarray(labels,dtype='uint8')
            (h,w,c) = data[0].shape
            x = np.ndarray((batch_size,h,w,c),dtype='uint8')
            for i in range(batch_size): x[i,:,:,:] = data[i]
        if norm: x = x/255.0
        yield(x,y)

def get_shapes(paths,gray_scale=True): #:::TO DO::: modify to use random sampling for faster result
    if gray_scale:
        ss = [tuple(list(cv2.imread(path,cv2.IMREAD_GRAYSCALE).shape)+[1]) for path in paths]
    else:
        ss = [cv2.imread(path).shape for path in paths]
    return list(set(ss))

def get_labels(paths,class_idx,offset=-1):
    labels = []
    for path in paths:
        lab = int(path.rsplit('label_')[-1].rsplit('/')[0])
        labels += [class_idx[lab]+offset]
    labels = np.asarray(labels,dtype='uint8')
    return labels

#only interested in when the labels don't match...
def confusion_matrix(pred_labels,test_labels,test_data=None,test_paths=None,
                     copy_error_dir=None,normalize=False,print_result=False,offset=1):
    M,m,n = {},set([]),0.0
    for i in test_labels:
        M[(i,i)] = 0.0
        for j in pred_labels:
            M[(i,j)] = 0.0
            M[(j,i)] = 0.0
            M[(j,j)] = 0.0
    for i in range(len(test_labels)):
        M[(test_labels[i],pred_labels[i])] += 1.0
        n += 1
        if copy_error_dir is not None and test_data is not None and test_paths is not None:
            out_path = copy_error_dir+'/%s_%s/'%(test_labels[i],pred_labels[i])
            if not os.path.exists(out_path): os.mkdir(out_path)
            base = test_paths.rsplit('/')[-1]
            cv2.imwrite(out_path+base,test_data[i])
    if normalize:
        for i,j in M: M[(i,j)] /= n
    if print_result:
        #total errors versus total correct
        ixs = {}
        for i,j in sorted(M,key=lambda x: (x[0],x[1])):
            if i in ixs: ixs[i] += [(i,j)]
            else:        ixs[i]  = [(i,j)]
        for i in ixs:
            print('\t'.join(['%s:%s'%((c[0]+offset,c[1]+offset),round(M[c],2)) for c in ixs[i]]))
    return M

def metrics(M,offset=1):
    ls,P,R,F1 = set([]),{},{},{}
    for i,j in M:
        if i+1 in P: P[i+offset] += [M[(i,j)]]
        else:        P[i+offset]  = [M[(i,j)]]
        if j+1 in R: R[j+offset] += [M[(i,j)]]
        else:        R[j+offset]  = [M[(i,j)]]
        ls.add(i)
        ls.add(j)
    for l in ls:
        sum_p = sum(P[l+offset])
        if sum_p>0.0:                   P[l+offset]  = M[(l,l)]/sum(P[l+offset])
        else:                           P[l+offset]  = 0.0
        sum_r = sum(R[l+offset])
        if sum_r>0.0:                   R[l+offset]  = M[(l,l)]/sum(R[l+offset])
        else:                           R[l+offset]  = 0.0
        if P[l+offset]+R[l+offset]>0.0: F1[l+offset] = 2.0*(P[l+offset]*R[l+offset])/(P[l+offset]+R[l+offset])
        else:                           F1[l+offset] = 0.0
    return P,R,F1

def plot_train_test(history,title='Model ACC+LOSS',ylim=[0.0,1.0],out_path=None,fontsize=8):
    ks   = history.keys()
    for k in ks:
        if k.find('val_')>=0 and k.find('val_loss')<0:
            ac = k.rsplit('val_')[-1]
            print('ploting %s'%ac)
    plt.plot(history[ac])
    plt.plot(history['val_%s'%ac])
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    axes = plt.gca()
    axes.set_ylim(ylim)
    plt.title(title,fontsize=fontsize)
    plt.ylabel('Accuracy & Loss Value')
    plt.xlabel('Epoch')
    plt.legend(['TRN-ACC','TST-ACC','TRN-LOSS','TST-LOSS'], loc='lower left')
    if out_path is not None: plt.savefig(out_path); plt.close()
    else: plt.show()
    return True

def plot_confusion_heatmap(confusion_matrix,title,offset=1,out_path=None,fontsize=8):
    xs = set([])
    for i,j in confusion_matrix:
        xs.add(i+offset)
        xs.add(j+offset)
    sx = sorted(list(xs))
    h = np.zeros((len(sx),len(sx)),dtype=float)
    for i,j in confusion_matrix:
        h[i,j] = confusion_matrix[(i,j)]
    plt.imshow(h,cmap='Greys')
    plt.xticks(range(len(sx)),sx)
    plt.yticks(range(len(sx)),sx)
    plt.title(title,fontsize=fontsize)
    plt.ylabel('Test Class')
    plt.xlabel('Pred Class')
    plt.colorbar()
    if out_path is not None: plt.savefig(out_path); plt.close()
    else: plt.show()
    return True
