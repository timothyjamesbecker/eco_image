#experiements for GOPRO image conversion to 600x200
import glob
import os
import cv2
import utils

in_dir  = '/media/tbecker/g_data/flow_data/_Video_Trail'
out_dir = '/media/tbecker/g_data/flow_data/_Video_Trail_Out'
if not os.path.exists(out_dir): os.mkdir(out_dir)
site_base  = '_Hiking_Trail_' #need to increment this...
gray_scale = False
equalize   = True
pixel_res  = [300,100]
n          = 5

j = 0
for image in sorted(glob.glob(in_dir+'/*.JPG')):
    j -= 1
    site_name = str(j)+site_base+image.split('/')[-1].split('.')[0]
    exif     = utils.read_exif_tags(image)
    raw_img  = cv2.imread(image)
    camera   = exif['Image Make']
    date     = exif['Image DateTime']
    seg_line = [0,int(raw_img.shape[0]*0.925),raw_img.shape[1],int(raw_img.shape[0]*0.925)]
    clip_img = utils.crop_seg(raw_img,seg_line)
    ml_imgs   = utils.multi_tilt_resize(clip_img,pixel_res[0],pixel_res[1],n=n,translate=True)
    for i in range(len(ml_imgs)):
        ml_img = ml_imgs[i]
        if gray_scale:
            ml_img = cv2.cvtColor(ml_img,cv2.COLOR_BGR2GRAY)
            ml_img = cv2.cvtColor(ml_img,cv2.COLOR_GRAY2BGR)
            if equalize: ml_img = cv2.equalizeHist(ml_img)
        else:
            if equalize: ml_img = utils.color_equalization(ml_img)
        out_file = out_dir+'/'+site_name+'_%s.JPG'%(i+1)
        cv2.imwrite(out_file,ml_img)