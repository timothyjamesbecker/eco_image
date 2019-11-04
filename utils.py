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
            if type(tags[t]) is not bytes:
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

def partition_data_paths(in_dir,split=0.15,enforce_test_site=False,verbose=True):
    L,C,LC,ls = {},{},{},set([])
    paths = sorted(glob.glob(in_dir+'/*/*.jpg')+glob.glob(in_dir+'/*/*.JPG'))
    for i in range(len(paths)):
        sid   = int(paths[i].rsplit('/')[-1].rsplit('_')[0])
        label = int(paths[i].rsplit('label_')[-1].rsplit('/')[0])
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
    if enforce_test_site: #makes a strict site holdout good to estimate overall performance on a new site
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
        train_sidx = sorted(list(set(sids).difference(set(test_sidx))))
        if verbose: print('sids %s selected for test cases'%(','.join([str(x) for x in test_sidx])))
        for sid in test_sidx:
            for [i,label] in L[sid]:
                tst_paths   += [paths[i]]
        for sid in train_sidx:
            for [i,label] in L[sid]:
                trn_paths  += [paths[i]]
    else: #take away split*(total_number of sites per label), makes a good estimate of classification confusion
        tst_paths,trn_paths,T = [],[],{}
        for l in LC:
            ts = max(1,round(len(LC[l])*split))
            test_sidx  = sorted(list(np.random.choice(LC[l],ts)))
            train_sidx = sorted(list(set(sids).difference(set(test_sidx))))
            for sid in test_sidx:
                for [i,label] in L[sid]:
                    tst_paths   += [paths[i]]
            for sid in train_sidx:
                for [i,label] in L[sid]:
                    trn_paths  += [paths[i]]
            T[l] = test_sidx
        trn_paths = sorted(list(set(trn_paths)))
        tst_paths = sorted(list(set(tst_paths)))
        if verbose:
            print('test sites randomly selected were:\n'+
                  '\n'.join(['%s:\t'%l+','.join([str(i) for i in T[l]]) for l in sorted(list(T.keys()))]))
    return trn_paths,tst_paths

def load_data_generator(paths,batch_size=64,gray_scale=True,norm=True,offset=-1):
    while True:
        data,labels,l = [],[],len(paths)
        idx = np.random.choice(range(l),size=min(batch_size,l),replace=False)
        if gray_scale:
            for i in idx:
                labels += [int(paths[i].rsplit('label_')[-1].rsplit('/')[0])+offset]
                data   += [cv2.imread(paths[i],cv2.IMREAD_GRAYSCALE)]
            y = np.asarray(labels,dtype='uint8')
            (h,w) = data[0].shape
            x = np.ndarray((batch_size,h,w,1),dtype='uint8')
            for i in range(batch_size): x[i,:,:,0] = data[i]
        else:
            for i in idx:
                labels += [int(paths[i].rsplit('label_')[-1].rsplit('/')[0])+offset]
                data   += [cv2.imread(paths[i])]
            y = np.asarray(labels,dtype='uint8')
            (h,w,c) = data[0].shape
            x = np.ndarray((batch_size,h,w,c),dtype='uint8')
            for i in range(batch_size): x[i,:,:,:] = data[i]
        if norm: x = x/255.0
        yield(x,y)

def get_shapes(paths,gray_scale=True):
    if gray_scale:
        ss = [tuple(list(cv2.imread(path,cv2.IMREAD_GRAYSCALE).shape)+[1]) for path in paths]
    else:
        ss = [cv2.imread(path).shape for path in paths]
    return list(set(ss))

def get_labels(paths,offset=-1):
    labels = []
    for path in paths:
        labels += [int(path.rsplit('label_')[-1].rsplit('/')[0])+offset]
    labels = np.asarray(labels,dtype='uint8')
    return labels

#only interested in when the labels don't match...
def confusion_matrix(pred_labels,test_labels,test_data=None,test_paths=None,
                     copy_error_dir=None,normalize=False,print_result=False,offset=1):
    M,m,n = {},set([]),0.0
    for i in test_labels:
        for j in pred_labels:
            M[(i,j)] = 0.0
    for i in range(len(test_labels)):
        if pred_labels[i] != test_labels[i]:
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

def plot_train_test(history,title='Model ACC+LOSS',ylim=[0.0,1.0],out_path=None):
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    axes = plt.gca()
    axes.set_ylim(ylim)
    plt.title(title)
    plt.ylabel('Accuracy & Loss Value')
    plt.xlabel('Epoch')
    plt.legend(['TRN-ACC','TST-ACC','TRN-LOSS','TST-LOSS'], loc='lower left')
    if out_path is not None: plt.savefig(out_path); plt.close()
    else: plt.show()
    return True

def plot_confusion_heatmap(confusion_matrix,title,offset=1,out_path=None):
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
    plt.title(title)
    plt.ylabel('Test Class')
    plt.xlabel('Pred Class')
    plt.colorbar()
    if out_path is not None: plt.savefig(out_path); plt.close()
    else: plt.show()
    return True
