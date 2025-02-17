#!/usr/bin/env python
import os
import time
import argparse
import glob
import random
import numpy as np
import cv2
import piexif
import pickle
import gzip
import multiprocessing as mp
import argparse
import utils

des="""
---------------------------------------------------
Bottom-Center Priority Image Crop/Resize Crawler
Timothy James Becker 10-19-19 to 02-28-20
---------------------------------------------------
Given input directory of images with EXIF metadata,
automatically performs an autocrop on the images
and non-destructively stores the result as a new
JPG file complete with EXIF tags or alternatively
as a numpy array with optional embedded EXIF tags
as a compressed pickle for use with ML workflows"""
parser = argparse.ArgumentParser(description=des.lstrip(" "),formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-i','--in_dir',type=str,help='input directory of images\t[None]')
parser.add_argument('-o','--out_dir',type=str,help='output directory\t[None]')
parser.add_argument('-p','--cpus',type=int,help='CPU cores to use for || processing\t[1]')
parser.add_argument('-r','--pixel_res',type=str,help='comma seperated horizontal,vertical pixel resolution\t[800,450]')
parser.add_argument('-a','--aug_bal',action='store_true',help='use rotation/slicing augmentation proportional to label frequency\t[False]')
parser.add_argument('-g','--gray_scale',action='store_true',help='use grayscale instead of color\t[False]')
args = parser.parse_args()

#build args into params...
if args.in_dir is not None:
    in_dir = args.in_dir
else:
    raise IOError
if args.out_dir is not None:
    out_dir = args.out_dir
    if not os.path.exists(out_dir): os.mkdir(out_dir)
else:
    raise IOError
if args.cpus is not None:
    cpus = args.cpus
else:
    cpus = 1
if args.pixel_res is not None:
    pixel_res = [int(x) for x in args.pixel_res.split(',')]
else:
    pixel_res = [800,450]

def label_spectrum(images):
    S = {}
    for image in images:
        label = int(image.split('label_')[-1].split('/')[0])
        if label not in S: S[label]  = 1
        else:              S[label] += 1
    return S

def balance_ratios(S):
    R = {}
    max_s = max([S[s] for s in sorted(S)])
    for s in S: R[s] = max_s/S[s]
    return R

result_list = [] #async queue to put results for || stages
def collect_results(result):
    result_list.append(result)

def crop_resize_images(images,out_dir,pixel_res,gray_scale,n_labels=None,equalize=True):
    out = {}
    for image in images:
        try:
            exif     = utils.read_exif_tags(image)
            raw_img  = cv2.imread(image)
            camera   = exif['Image Make']
            date     = exif['Image DateTime']
            seg_line = [0,int(raw_img.shape[0]*0.925),raw_img.shape[1],int(raw_img.shape[0]*0.925)]
            clip_img = utils.crop_seg(raw_img,seg_line)
            if n_labels is not None: #will be using data rotation and translation augmentation
                label = int(image.split('label_')[-1].split('/')[0])
                if n_labels[label]>1:
                    if label<=2:
                        ml_imgs = utils.multi_tilt_resize(clip_img,pixel_res[0],pixel_res[1],n_labels[label],deg=30,translate=True)
                    else:
                        ml_imgs = utils.multi_tilt_resize(clip_img,pixel_res[0],pixel_res[1],n_labels[label],deg=30)
                else:
                    ml_imgs = [utils.resize(clip_img,pixel_res[0],pixel_res[1])]
                x = 1
                for ml_img in ml_imgs:
                    if gray_scale:
                        ml_img = cv2.cvtColor(ml_img,cv2.COLOR_BGR2GRAY)
                        ml_img = cv2.cvtColor(ml_img,cv2.COLOR_GRAY2BGR)
                        if equalize: ml_img = cv2.equalizeHist(ml_img)
                    else:
                        if equalize: ml_img = utils.color_equalization(ml_img)
                    if not os.path.exists(out_dir+('/label_%s'%label)): os.mkdir(out_dir+('/label_%s'%label))
                    out_file = out_dir+('/label_%s'%label)+'/'+\
                               image.split('/')[-1].replace(' ','_').replace('.JPG','').replace('.jpg','')+'_AUG%s.JPG'%x
                    cv2.imwrite(out_file,ml_img)
                    piexif.insert(piexif.dump(piexif.load(image)),out_file)
                    x += 1
            else:
                ml_img   = utils.resize(clip_img,pixel_res[0],pixel_res[1])
                if gray_scale:
                    ml_img = cv2.cvtColor(ml_img,cv2.COLOR_BGR2GRAY)
                    ml_img = cv2.cvtColor(ml_img,cv2.COLOR_GRAY2BGR)
                    if equalize: ml_img = cv2.equalizeHist(ml_img)
                else:
                    if equalize: ml_img = utils.color_equalization(ml_img)
                if len(image.split('/'))>2 and image.split('/')[-2].startswith('label_'):
                    label = image.split('label_')[-1].split('/')[0]
                    if not os.path.exists(out_dir+'/label_'+label): os.mkdir(out_dir+'/label_'+label)
                    out_file = out_dir+'/label_'+label+'/'+image.split('/')[-1]
                    cv2.imwrite(out_file,ml_img)
                else:
                    out_file = out_dir+'/'+image.split('/')[-1]
                    cv2.imwrite(out_file,ml_img)
                piexif.insert(piexif.dump(piexif.load(image)),out_file)
            out[image] = True
        except Exception as E:
            out[image] = E
            pass
    return out

if __name__ == '__main__':
    images = glob.glob(in_dir+'/*/*.jpg')+glob.glob(in_dir+'/*/*.JPG')+glob.glob(in_dir+'/*.jpg')+glob.glob(in_dir+'/*.JPG')
    n = int(len(images)//cpus)
    image_sets = [images[i*n:(i+1)*n] for i in range(cpus)]
    if len(images)%cpus>0: image_sets[-1] += [images[-1]]
    print('using pixel resolution:%sx%s for %s image sets of average size:%s'%(pixel_res[0],pixel_res[1],len(image_sets),n))

    if args.aug_bal:
        print('calculating label spectrum to employ balanced augmentation processing...')
        S = label_spectrum(images)
        for s in sorted(S): print('label=%s:%s'%(s,S[s]))
        R = balance_ratios(S)
        n_labels = {r:int(round(min(R[r],3),0)) for r in sorted(R)}
        print(n_labels)

    else: n_labels = None
    start = time.time()
    p1 = mp.Pool(cpus)
    for image_subset in image_sets:  # each site in ||
        p1.apply_async(crop_resize_images,
                       args=(image_subset,out_dir,pixel_res,args.gray_scale,n_labels,True),
                       callback=collect_results)
        time.sleep(0.1)
    p1.close()
    p1.join()
    stop  = time.time()
    print('processed %s images in %s sec using %s cpus'%(len(images), round(stop-start,2),cpus))
    print('or %s images per sec'%(len(images)/(stop-start)))