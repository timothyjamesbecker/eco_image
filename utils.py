import os
import sys
import exifread
import piexif
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

def local_path():
    return '/'.join(os.path.abspath(__file__).split('/')[:-1])+'/'

#given an image file with exif metadat return set of the tags that are required
def read_exif_tags(path,tag_set='all'):
    tags,T = {},{}
    with open(path,'rb') as f: tags = exifread.process_file(f)
    if tag_set=='all': tag_set = set(list(tags.keys()))
    for t in sorted(list(tags.keys())):
        if t in tag_set and type(tags[t]) is not str:
            tag_value = tags[t].values
            if type(tag_value) is list: tag_value = ','.join([str(v) for v in tag_value])
            if type(tag_value) is str: tag_value = tag_value.rstrip(' ')
            T[t] = str(tag_value)
    return T

def set_exif_tags(path,tag_set):
    return True

def get_camera_seg_mult(camera_type):
    if camera_type.upper().startswith('MOULTRIE'):
        offset = 0.10
    elif camera_type.upper().startswith('BUSHNELL'):
        offset = 0.925
    elif camera_type.upper().startswith('SPYPOINT'):
        offset = 0.75
    else:
        offset = 0.5
    return offset

#uses the
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

#seg is one of four orientations t- r| b- |l img needs a cut first
def crop_seg(img,seg):
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
        img = cv2.resize(img,dim,interpolation=interp)
        d = int(round((img.shape[1]-width)/2.0))
        img = img[:,d:(img.shape[1]-d),:]
    else:
        dim = (int(round(img.shape[1]*w_scale)),
               int(round(img.shape[0]*w_scale)))
        img = cv2.resize(img,dim,interpolation=interp)
        d = int(round((img.shape[0]-height)))
        img = img[d:img.shape[0],:,:]
    img = cv2.resize(img,(width,height),interpolation=interp)
    return img

def plot(img):
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
        test_data    = np.ndarray(shape=(len(test_idx),h,w),dtype='uint8')
        train_data   = np.ndarray(shape=(len(train_idx),h,w),dtype='uint8')
        for i in range(len(test_data)): test_data[i,:,:]  = data[test_idx[i]]
        for i in range(len(train_data)):train_data[i,:,:] = data[train_idx[i]]
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
def load_data_advanced(in_dir,gray_scale=True,split=0.15,verbose=True):
    L = {}
    paths = sorted(glob.glob(in_dir+'/*/*.jpg')+glob.glob(in_dir+'/*/*.JPG'))
    for i in range(len(paths)):
        sid   = int(paths[i].rsplit('/')[-1].rsplit('_')[0])
        label = int(paths[i].rsplit('label_')[-1].rsplit('/')[0])
        if sid in L: L[sid] += [[i,label]]
        else:        L[sid]  = [[i,label]]
    trn_data,trn_labels,tst_data,tst_labels = [],[],[],[]
    ts   = max(1,round(len(L)*split))
    sids = list(L.keys())
    test_sidx  = sorted(list(np.random.choice(sids,ts)))
    train_sidx = sorted(list(set(sids).difference(set(test_sidx))))
    if verbose: print('sids %s selected for test cases'%(','.join([str(x) for x in test_sidx])))
    if gray_scale:
        for sid in test_sidx:
            for [i,label] in L[sid]:
                tst_labels  += [label]
                tst_data    += [cv2.imread(paths[i],cv2.IMREAD_GRAYSCALE)]
        for sid in train_sidx:
            for [i,label] in L[sid]:
                trn_labels += [label]
                trn_data   += [cv2.imread(paths[i],cv2.IMREAD_GRAYSCALE)]
        (h,w) = trn_data[0].shape
        train_labels = np.asarray(trn_labels,dtype='uint8')
        test_labels  = np.asarray(tst_labels,dtype='uint8')
        train_data   = np.ndarray((len(trn_data),h,w),dtype='uint8')
        test_data    = np.ndarray((len(tst_data),h,w),dtype='uint8')
        for i in range(len(train_data)): train_data[i,:,:] = trn_data[i]
        for i in range(len(test_data)):  test_data[i,:,:]  = tst_data[i]
    else:
        for sid in test_sidx:
            for [i,label] in L[sid]:
                tst_labels  += [label]
                tst_data    += [cv2.imread(paths[i])]
        for sid in train_sidx:
            for [i,label] in L[sid]:
                trn_labels += [label]
                trn_data   += [cv2.imread(paths[i])]
        (h,w,c) = trn_data[0].shape
        train_labels = np.asarray(trn_labels,dtype='uint8')
        test_labels  = np.asarray(tst_labels,dtype='uint8')
        train_data   = np.ndarray((len(trn_data),h,w,c),dtype='uint8')
        test_data    = np.ndarray((len(tst_data),h,w,c),dtype='uint8')
        for i in range(len(train_data)): train_data[i,:,:,:] = trn_data[i]
        for i in range(len(test_data)):  test_data[i,:,:,:]  = tst_data[i]
    return (train_data,train_labels),(test_data,test_labels)
