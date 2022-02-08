import os
import glob
import cv2
import time
import json
import datetime as dt
import numpy as np
import multiprocessing as mp
import argparse
import utils

def get_sid(path):
    return int(path.split('/')[-1].split('_')[0])

def get_label(path):
    if path.find('label_')>-1:
        label = int(path.split('label_')[-1].split('/')[0])
    else:
        label = 0
    return label

def get_deploy(path):
    return '_'.join(path.split('/')[-1].split('_')[2:5]).split(' ')[0]

def get_img_num(path):
    return int(path.split('.JPG')[0].split('_')[-1].split(' ')[-1].replace('(','').replace(')',''))

#read all images exif data to order and partition triggered events,returns: order,error,trigs
def temporally_order_paths(S,time_bin=5):
    O = {}
    for sid in S:
        O[sid] = {}
        for deploy in sorted(S[sid]):
            O[sid][deploy] = {'order':[],'exif':[],'trigs':[]}
            start = dt.datetime.strptime(deploy.split('_')[0],'%m%d%y')-dt.timedelta(seconds=1)
            stop  = dt.datetime.strptime(deploy.split('_')[1],'%m%d%y')+dt.timedelta(hours=24)
            #---------------------------------------------------------------------------------
            raw = []
            for i in range(len(S[sid][deploy])):
                path  = S[sid][deploy][i]
                label = get_label(path)
                exif  = utils.read_exif_tags(path)
                ct    = exif['Image Make']
                ts    = dt.datetime.strptime(exif['Image DateTime'],'%Y:%m:%d %H:%M:%S')
                if ts>=start and ts<=stop:
                    raw += [[ts,label,ct,path]]
                else:
                    O[sid][deploy]['exif'] += [[ts,label,ct,path]]
            if len(raw)<1 and len(O[sid][deploy]['exif'])>1: #try to fix the year?
                print('exif date stamps do not match deployment dates for %s images...'%(len(O[sid][deploy]['exif'])))
                exif_issues,corrected = O[sid][deploy]['exif'],[]
                start_e = np.min([e[0] for e in exif_issues])
                stop_e  = np.max([e[0] for e in exif_issues])
                year_diff = int(np.round(np.mean([(start-start_e).total_seconds()/(60*60*24*365),
                                                  (stop - stop_e).total_seconds()/(60*60*24*365)])))
                print('attempting to correct exif date stamps using offset=%s years'%year_diff)
                for i in range(len(exif_issues)):
                    ts = exif_issues[i][0]
                    nt = ts+dt.timedelta(days=365*year_diff)
                    if nt>=start and nt<=stop: #does the corrected date fix the issues?
                        corrected += [i]       #save its index so we can still toss some...
                        exif_issues[i][0] = nt #patch the corrected year into the datetime stamp
                issues = sorted(set(range(len(exif_issues))).difference(set(corrected)))
                print('%s corrections were made, %s images remain with unfixable issues...'%(len(corrected),len(issues)))
                O[sid][deploy]['exif'] = []
                for i in issues:    O[sid][deploy]['exif'] += [exif_issues[i]]
                for c in corrected: raw += [exif_issues[c]]
            raw = sorted(raw, key=lambda x: x[0])

            #[0] find the maximal deployment timestamp point (within the hour): camera capture rate and offset
            deploy_days = (stop-start).days
            deploy_hours = deploy_days*24    #maximal number of temporal triggered events
            T = [dt.timedelta(minutes=int(t)) for t in np.arange(0,time_bin+60,time_bin)]
            H = {t:0 for t in T}
            for r in raw:
                _t = dt.timedelta(minutes=r[0].time().minute,seconds=r[0].time().second)
                for t in range(1,len(T),1):
                    if _t>=T[t-1] and _t<T[t]:
                        H[T[t-1]] += 1
                        break
            max_ts = [dt.timedelta(0),0]
            for h in H:
                if H[h]>max_ts[1]: max_ts = [h,H[h]]

            #[1] filter out timestamps in the non-maximal point
            R = []
            for r in raw:
                _t = dt.timedelta(minutes=r[0].time().minute,seconds=r[0].time().second)
                if _t>=max_ts[0] and _t<max_ts[0]+dt.timedelta(minutes=time_bin):
                    R += [_t]
            if len(R)>0:
                mean_ts = np.mean(R)
                for r in raw:
                    _t = dt.timedelta(minutes=r[0].time().minute,seconds=r[0].time().second)
                    if abs(mean_ts-_t).total_seconds()<time_bin*60.0: O[sid][deploy]['order']   += [r]
                    else:                                               O[sid][deploy]['trigs'] += [r]
            n_error,n_trig,n_tot = len(O[sid][deploy]['exif']),len(O[sid][deploy]['trigs']),len(S[sid][deploy])
            if n_error>0:
                print('sid=%s,deploy=%s: %s or %s/%s images had timestamps outside the deployment...'\
                      %(sid,deploy,round(n_error/n_tot,2),n_error,n_tot))
            if n_trig>0:
                print('sid=%s,deploy=%s: %s or %s/%s images were potential triggered events...'\
                      %(sid,deploy,round(n_trig/n_tot,2),n_trig,n_tot))
    return O

#given a sid and deployment (label), look for events: (chroma dropped, camera moved, lens flare, moisture blur)
def detect_events(imgs,ref,min_size=300):
    x,events = 1,{'dark':{},'light':{},'bw':{},'rotated':{},'blurred':{},'flared':{}}
    if len(imgs)>0:
        if imgs[0].shape[0]>min_size:
            while imgs[0].shape[0]//x>min_size: x+=1
    for i in imgs:
        if x>1:
            imgA = utils.resize(imgs[i],imgs[i].shape[1]//x,imgs[i].shape[0]//x)
            refA = utils.resize(ref,ref.shape[1]//x,ref.shape[0]//x)
        else:
            imgA = imgs[i]
            refA = ref
        bw  = utils.chroma_dropped(imgA)
        drk = utils.too_dark(imgA)
        lht = utils.too_light(imgA)
        blr = utils.blurred(imgA)
        flr = utils.lens_flare(imgA)
        rot = utils.luma_rotated(refA,imgA)
        if bw:  events['bw'][i]      = True
        if drk: events['dark'][i]    = True
        if lht: events['light'][i]   = True
        if rot: events['rotated'][i] = True
        if blr: events['blurred'][i] = True
        if flr: events['flared'][i]  = True
    return events

#returns the ref image of the first good ref image: (color, sharp, etc..)
def skip_to_ref(paths,width,height,verbose=True):
    i,ref = 0,None
    if len(paths)>0:
        ref = utils.read_crop_resize(paths[0],height=height,width=width)
        while i<len(paths) and utils.chroma_dropped(ref): #will find the first one that meets all the checks...
            ref = utils.read_crop_resize(paths[i],height=height,width=width)
            i += 1
        if verbose: print('skipped past %s ref images in sid=%s, deploy=%s'%(i,sid,deploy))
    return ref

def pad_imgs(raw_imgs,rot,width,height,pad):
    pads = {}
    if len(rot)>0:
        pads = {i:utils.get_rotation_pad(raw_imgs[i]) for i in sorted(set(rot).intersection(set(raw_imgs)))}
        if len(pads)>0:
            max_pad,avg_pad = max([pads[i] for i in pads]),int(round(sum([pads[i] for i in pads])/len(pads)))
            if max_pad<min(height*pad,width*pad):
                for i in sorted(raw_imgs):
                    raw_imgs[i] = utils.resize(utils.crop(raw_imgs[i],pixels=max_pad),height=height,width=width)
    return raw_imgs

def apply_homography(ref,raw_imgs,rot,pixel_dist=100):
    for i in raw_imgs:
        if i in rot:
            img,hmg  = utils.homography(ref,raw_imgs[i],pixel_dist=pixel_dist)
            raw_imgs[i] = img
    return raw_imgs

#imgs is a dictionary with potentially missing keys (due to events)
def temporal_nn_imgs(imgs,events,ts,hours=4): #ts has all indexes
    NN,n = {},len(ts)
    for i in sorted(ts):
        NN[i] = []
        l,r = i-1,i+1
        while l>0 and ts[l].date()==ts[i].date() and np.abs(ts[i]-ts[l]).total_seconds()<=(hours*60*60):
            if l in imgs and l not in events['bw']: NN[i] += [l]
            l-=1
        while r<n and ts[r].date()==ts[i].date() and np.abs(ts[i]-ts[r]).total_seconds()<=(hours*60*60):
            if r in imgs and r not in events['bw']: NN[i] += [r]
            r+=1
        NN[i] = sorted(NN[i])
    return NN

#returns the index of the image from a list that has the smallest luma differential to the target
def central_image(target,imgs): #imgs is a list here...
    min_i = [None,None]
    if len(imgs)>0:
        min_i= [0,np.sum(utils.luma_diff(target,imgs[0]))]
        for i in range(len(imgs)):
            luma_diff = np.sum(utils.luma_diff(target,imgs[i]))
            if luma_diff<=min_i[1]: min_i = [i,luma_diff]
    return min_i

#aggregation function splits a day into sections (~12 daylight hours / aggregate_hours = result hours
#clusters the avaible timestamps by the agregation paramter: hours
def select_aggregate_imgs(imgs,ts,ls=None,hours=1,composite=True): #for one sid and deploy setup...
    D = {}
    for i in imgs:
        d = ts[i].date()
        if ls is not None and len(set(ts).difference(set(ls)))<=0:
            if d in D: D[d] += [[ts[i],ls[i],imgs[i]]] #0 is the default label
            else:      D[d]  = [[ts[i],ls[i],imgs[i]]] #0 is the default label
        else:
            if d in D: D[d] += [[ts[i],0,imgs[i]]] #0 is the default label
            else:      D[d]  = [[ts[i],0,imgs[i]]] #0 is the default label
    for d in sorted(D): D[d] = sorted(D[d],key=lambda x: x[0])
    A = {}
    for d in sorted(D): #each day
        A[d] = {}
        min_ts,max_ts = np.min([x[0] for x in D[d]]),np.max([x[0] for x in D[d]])
        range_ts = np.round(np.abs(max_ts-min_ts).total_seconds()/(60*60))
        c_ts,CL = [],{} #time stamp clusters for unit that is hours long
        for i in range(int(range_ts//hours)+1):
            c = min_ts+dt.timedelta(hours=i*hours)
            c_ts += [c]
            CL[c] = []
        c_ts += [dt.datetime.combine(max_ts,dt.time.max)]
        for i in range(len(D[d])):
            d_t = D[d][i][0]
            for t in range(0,len(c_ts)-1,1):
                if d_t>=c_ts[t] and d_t<c_ts[t+1]:
                    CL[c_ts[t]] += [[D[d][i][-1],D[d][i][1]]]
        for c in CL:
            img_c,img_l = [e[0] for e in CL[c]],[e[1] for e in CL[c]]
            sharpness = (sharp if len(img_c)>10 else sharp-sharp/len(img_c))
            hmean = utils.hue_mean(img_c)
            smean = utils.sat_mean(img_c)
            if sharpness>0.0: vmean = utils.sharpen(utils.val_mean(img_c),amount=sharp)
            else:             vmean = utils.val_mean(img_c)
            color = np.zeros_like(img_c[0])
            color[:,:,0] = hmean
            color[:,:,1] = smean
            color[:,:,2] = vmean
            color = cv2.cvtColor(color,cv2.COLOR_HSV2BGR)
            label = np.mean(img_l)
            if not composite: A[d][c] = CL[c][central_image(color,CL[c])]
            else:             A[d][c] = [color,label]
    return A

def process_image_partitions(C,params):
    width,height         = params['width'],params['height']
    enh_hrs,agg_hrs      = params['enh_hrs'],params['agg_hrs']
    equalize,advanced    = params['equalize'],params['advanced']
    mean,sharp,winsize   = params['mean'],params['sharp'],params['winsize']
    pad,d_min,pixel_dist = params['pad'],params['d_min'],params['pixel_dist']
    write_labels         = params['write_labels']
    for sid in C:
        for deploy in sorted(C[sid]):
            n = len(C[sid][deploy])
            print('processing %s paths for sid=%s, deploy=%s'%(n,sid,deploy))

            #[1] find a starting reference image point::::::::::::::::::::::::::::::::::::::::::::::::::::::::
            ref = skip_to_ref([e[-1] for e in C[sid][deploy]],width=width,height=height) #[datetime,label,camera,path]

            #[2] read images and detect image events::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
            raw_imgs = {}
            for i in range(n):
                raw_imgs[i] = utils.read_crop_resize(C[sid][deploy][i][-1],height=height,width=width)

            #[3] find events and partition the images on quality::::::::::::::::::::::::::::::::::::::::
            events = detect_events(raw_imgs,ref)
            ls = {i:C[sid][deploy][i][1] for i in range(len(C[sid][deploy]))} #get original labels if they exist
            sharp_imgs = {}
            for i in raw_imgs:
                if i in ls: label = ls[i]
                else:       label = 0
                if i not in events['bw']:
                    if i not in events['blurred']:
                        if i not in events['flared']:
                            if i not in events['dark']:
                                if i not in events['light']:
                                    sharp_imgs[i] = raw_imgs[i] #passed filters
                                else:
                                    light_dir = out_dir+'/light'
                                    if not os.path.exists(light_dir): os.mkdir(light_dir)
                                    if write_labels: img_name = light_dir+'/S%s_D%s_I%s_L%s.JPG'%(sid,deploy,i+1,label)
                                    else:            img_name = light_dir+'/S%s_D%s_I%s.JPG'%(sid,deploy,i+1)
                                    cv2.imwrite(img_name,raw_imgs[i])
                                    print('LLL sid=%s, deploy=%s: image %s was too light LLL'%(sid,deploy,i))
                            else:
                                dark_dir = out_dir+'/dark'
                                if not os.path.exists(dark_dir): os.mkdir(dark_dir)
                                if write_labels: img_name = dark_dir+'/S%s_D%s_I%s_L%s.JPG'%(sid,deploy,i+1,label)
                                else:            img_name = dark_dir+'/S%s_D%s_I%s.JPG'%(sid,deploy,i+1)
                                cv2.imwrite(img_name,raw_imgs[i])
                                print('DDD sid=%s, deploy=%s: image %s was too dark DDD'%(sid,deploy,i))
                        else:
                            flared_dir = out_dir+'/flared'
                            if not os.path.exists(flared_dir): os.mkdir(flared_dir)
                            if write_labels: img_name = flared_dir+'/S%s_D%s_I%s_L%s.JPG'%(sid,deploy,i+1,label)
                            else:            img_name = flared_dir+'/S%s_D%s_I%s.JPG'%(sid,deploy,i+1)
                            cv2.imwrite(img_name,raw_imgs[i])
                            print('FFF sid=%s, deploy=%s: image %s was too flared FFF'%(sid,deploy,i))
                    else:
                        blurred_dir = out_dir+'/blurred'
                        if not os.path.exists(blurred_dir): os.mkdir(blurred_dir)
                        if write_labels: img_name = blurred_dir+'/S%s_D%s_I%s_L%s.JPG'%(sid,deploy,i+1,label)
                        else:            img_name = blurred_dir+'/S%s_D%s_I%s.JPG'%(sid,deploy,i+1)
                        cv2.imwrite(img_name,raw_imgs[i])
                        print('BBB sid=%s, deploy=%s: image %s was too blurred BBB'%(sid,deploy,i))
                else:
                    bw_dir = out_dir+'/bw'
                    if not os.path.exists(bw_dir): os.mkdir(bw_dir)
                    if write_labels: img_name = bw_dir+'/S%s_D%s_I%s_L%s.JPG'%(sid,deploy,i+1,label)
                    else:            img_name = bw_dir+'/S%s_D%s_I%s.JPG'%(sid,deploy,i+1)
                    cv2.imwrite(img_name,raw_imgs[i])
                    print('BW sid=%s, deploy=%s: image %s was chroma dropped BW'%(sid,deploy,i))

            #[4] apply fixes for those that can be fixed::::::::::::::::::::::::::::::::::::::::::::::::::::
            trans_imgs = apply_homography(ref,sharp_imgs,events['rotated'],pixel_dist)
            imgs       = pad_imgs(trans_imgs,events['rotated'],width=width,height=height,pad=pad)
            print('preprocessed %s images (partitioned on dark,light,b&w,rotation,blur events)...'%len(imgs))

            #[5] complete temporal clusterings and final luma+, hue+ enhancements:::::::::::::::::::::::::::::::::::::::
            ts = {i:C[sid][deploy][i][0] for i in range(len(C[sid][deploy]))}
            NN = temporal_nn_imgs(imgs,events,ts,hours=enh_hrs)
            if agg_hrs<=1: #have regular potential enhanced images here, lets apply labels if they exist and
                for i in imgs:
                    if i in ls: label = ls[i]
                    else:       label = 0
                    if len(NN[i])>0:
                        lmean     = utils.sharpen(utils.luma_mean([imgs[x] for x in NN[i]],equalize),amount=sharp)
                        luma      = utils.luma_enhance(imgs[i],lmean,amount=mean,winsize=winsize,advanced=advanced)
                        hmean     = utils.hue_mean([imgs[x] for x in NN[i]])
                        smean     = utils.sat_mean([imgs[x] for x in NN[i]])
                        hue       = utils.color_enhance(luma,hmean,smean,amount=mean)

                        passed_dir = out_dir+'/passed'
                        if not os.path.exists(passed_dir): os.mkdir(passed_dir)
                        # img_name = passed_dir+'/S%s_D%s_I%s_LE_L%s.JPG'%(sid,deploy,i,label)
                        # cv2.imwrite(img_name,luma)
                        if write_labels: img_name = passed_dir+'/S%s_D%s_I%s_HE_L%s.JPG'%(sid,deploy,i,label)
                        else:            img_name = passed_dir+'/S%s_D%s_I%s_HE.JPG'%(sid,deploy,i)
                        cv2.imwrite(img_name,hue)
                        # img_name = passed_dir+'/S%s_D%s_I%s_N.JPG'%(sid,deploy,i)
                        # cv2.imwrite(img_name,imgs[i])
                    else:
                        passed_dir = out_dir+'/passed'
                        if not os.path.exists(passed_dir): os.mkdir(passed_dir)
                        if write_labels: img_name = passed_dir+'/S%s_D%s_I%s_NE_L%s.JPG'%(sid,deploy,i,label)
                        else:            img_name = passed_dir+'/S%s_D%s_I%s_NE%s.JPG'%(sid,deploy,i)
                        cv2.imwrite(img_name,imgs[i])
                        print('can not enhance img=%s,sid=%s,deploy=%s'%(i,sid,deploy))
            else: #aggregation of the images will generate a JSON mapping file...
                print('aggregation hours is > 1, proceeding to aggregate images by %s hours'%agg_hrs)
                weekday = {0:'Mon',1:'Tues',2:'Wed',3:'Thur',4:'Fri',5:'Sat',6:'Sun'}
                AL = select_aggregate_imgs(imgs,ts,ls=ls,hours=agg_hrs)
                for d in sorted(AL):
                    date_str = d.strftime('%y-%m-%d')
                    weekday_str = weekday[d.weekday()]
                    passed_dir = out_dir+'/passed_aggregated'
                    if not os.path.exists(passed_dir): os.mkdir(passed_dir)
                    for c in sorted(AL[d]):
                        time_str = c.strftime('%H')
                        img_name = passed_dir+'/S%s_D%s_DATE%s_%s_%s_L%s.JPG'%\
                                   (sid,deploy,date_str,weekday_str,time_str,round(AL[d][c][1],2))
                        cv2.imwrite(img_name,AL[d][c][0])
    return True

def generate_img_map_json(out_dir,json_path,row_type='array'): #puts together metadata for an image datsets including week enumeration
    meta,weekday = [],{0:'Mon',1:'Tues',2:'Wed',3:'Thur',4:'Fri',5:'Sat',6:'Sun'}
    for full_img_path in glob.glob(out_dir+'/passed*/*.JPG'):
        img_path = full_img_path.split('/')[-1]
        sid      = int(img_path.split('S')[1].split('_')[0])
        deploy   = img_path.split('D')[1].rstrip('_')
        im_date  = dt.datetime.strptime(img_path.split('DATE')[-1].split('_')[0],'%y-%m-%d')
        wkday    = img_path.split('DATE')[-1].split('_')[1]
        hr       = int(img_path.split('DATE')[-1].split('_')[2])
        label    = float(img_path.split('L')[-1].split('.JPG')[0])
        meta += [[sid,deploy,im_date,wkday,hr,img_path,label]]
    meta = sorted(meta,key=lambda x: x[2])

    #find the first Monday, and the last... for week enumeration
    min_day,max_day = meta[0][2],meta[-1][2]
    while weekday[min_day.weekday()]!='Mon':
        min_day -= dt.timedelta(days=1)
    while weekday[max_day.weekday()]!='Mon':
        max_day -= dt.timedelta(days=1)
    weeks,w = [],min_day
    while w<max_day:
        weeks += [w]
        w += dt.timedelta(days=7)
    W = {}
    for m in meta:
        for w in range(1,len(weeks)-1,1):
            if m[2]>=weeks[w] and m[2]<weeks[w+1]:
                if weeks[w] in W: W[weeks[w]] += [m]
                else:             W[weeks[w]]  = [m]
    n,N = 1,{} #enumerated data week number
    if row_type=='object':
        for w in sorted(W):
            N[n] = {'weekstart_date':w.strftime('%y-%m-%d'),'data':{}}
            for i in range(len(W[w])):
                sid,deploy,img_date,wk_day,hr,img_path,label = W[w][i]
                img_date = img_date.strftime('%y-%m-%d')
                row = {'deploy':deploy,'image':img_path,'label':label,
                       'date':img_date,'hour':hr,'weekday':wk_day}
                if sid in N[n]['data']: N[n]['data'][sid] += [row]
                else:                   N[n]['data'][sid]  = [row]
            N[n]['data'][sid] = sorted(N[n]['data'][sid],key=lambda x: x['date'])
            n += 1
    elif row_type=='array':
        for w in sorted(W):
            N[n] = {'weekstart_date':w.strftime('%y-%m-%d'),'data':{}}
            for i in range(len(W[w])):
                sid,deploy,img_date,wk_day,hr,img_path,label = W[w][i]
                img_date = img_date.strftime('%y-%m-%d')
                row = [deploy,img_path,label,img_date,hr,wk_day]
                if sid in N[n]['data']: N[n]['data'][sid] += [row]
                else:                   N[n]['data'][sid]  = [row]
            N[n]['data'][sid] = sorted(N[n]['data'][sid],key=lambda x: x[3])
            n += 1
    with open(json_path,'w') as f:
        json.dump(N,f)
        return True
    return False

result_list = [] #async queue to put results for || stages
def collect_results(result):
    result_list.append(result)

#this will become an arg.parse command line with a main method......................................
#it will replace the basic downsample_crawler.py and be called: raw_image_prep.py
if __name__ == '__main__':
    des="""
    ------------------------------------------------------------
    Stream/River Image Processor (SRIP)
    
    -Bottom-Center Auto-Cropping (maximal interpolated resizing)
    -Temporal k-NN luma and chroma enhancement
    -Over,Under,B&W,Flared,Blurred,Rotated Detection
    
    (c) Timothy James Becker 10-19-19 to 02-06-22
    ------------------------------------------------------------
    Given input directory of temporal stream images with EXIF metadata,
    detects and partitions images into usable and unusable folders.
    Images that pass the event detectors are uniformly resized
    and bottom-center cropped (to attempt to put water in the same position
    in each frame).  Additional luma and chrom enhancement can be applied
    to provide more stable images with less speculiar highlghts and shadows
    that are common with forested environments"""
    parser = argparse.ArgumentParser(description=des.lstrip(" "),formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--in_dir',type=str,help='input directory of images\t[None]')
    parser.add_argument('--out_dir',type=str,help='output directory prefix\t[None]')
    parser.add_argument('--sids',type=str,help='specific comma seperated sids to process\t[all]')
    parser.add_argument('--res',type=str,help='comma seperated output width,height\t[600,200]')
    parser.add_argument('--mean',type=float,help='luma and hue mean applied during enhancement\t[1.0]')
    parser.add_argument('--sharp',type=float,help='sharpness applied after luma before enhancement\t[2.0]')
    parser.add_argument('--enh_hrs',type=int,help='temporal based NN enhancement hours\t[2]')
    parser.add_argument('--agg_hrs', type=int, help='number of hours per day to aggregate\t[2]')
    parser.add_argument('--equalize',action='store_true',help='luma channel histogram equalization\t[False]')
    parser.add_argument('--json_img_map',action='store_true',help='generate a temporal weekly json image map\t[False]')
    parser.add_argument('--label_sub_folders',action='store_true',help='generate subfolders for passed images\t[False]')
    parser.add_argument('--advanced',action='store_true',help='luma differential corrected by dense optical flow\t[False]')
    parser.add_argument('--winsize',type=int,help='kernel window for optical flow if using advanced\t[2]')
    parser.add_argument('--cpus',type=int,help='CPU cores to use for || processing\t[1]')
    args = parser.parse_args()

    if args.in_dir is not None:
        in_dir = args.in_dir
    else: raise IOError
    if args.out_dir is not None:
        out_dir = args.out_dir
    else: raise IOError
    if args.sids is not None:
        sids = [int(sid) for sid in args.sids.split(',')]
    if args.res is not None:
        width,height = [int(r) for r in args.res.split(',')]
    if args.mean is not None:
        mean = args.mean
    else: mean = 1.0
    if args.sharp is not None:
        sharp = args.sharp
    else: sharp = 2.0
    if args.enh_hrs is not None:
        enh_hrs = args.enh_hrs
    else: enh_hrs = 0
    if args.agg_hrs is not None:
        agg_hrs = args.agg_hrs
    else: agg_hrs = 0
    if args.equalize:
        equalize = True
    else: equalize = False
    if args.advanced:
        advanced = True
    else: advanced = False
    if args.winsize is not None:
        winsize = args.winsize
    else: winsize = 5
    if args.cpus is not None:
        cpus = args.cpus
    else: cpus = 1

    #sids = [19022]#,14434,14523,15244]

    params = {'width':width,'height':height,'enh_hrs':enh_hrs,'agg_hrs':agg_hrs,'equalize':equalize,'advanced':advanced,
              'mean':mean,'sharp':sharp,'winsize':winsize,'pad':0.2,'d_min':2.0,'pixel_dist':4*height//10,
              'write_labels':args.label_sub_folders}
    print('using params:%s'%params)
    raw_path = in_dir+'/*/*.JPG'
    raw_paths = sorted(glob.glob(raw_path),key=get_sid)
    if len(raw_paths)<0:
        print('raw image data is not ordered by label, proceeding to locate unlabeled JPGs...')
        raw_paths = sorted(glob.glob(in_dir+'/*.JPG'))
        print('located %s unlabeled image paths'%(len(raw_paths)))
    else: print('located %s label ordered image paths'%len(raw_paths))

    while out_dir[-1]=='/': out_dir = out_dir[:-1]
    out_dir = out_dir+'_ehrs%s_ahrs%s_%s/'%(params['enh_hrs'],params['agg_hrs'],('advanced' if params['advanced'] else 'basic'))
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    print('output directory has been created at:%s'%out_dir)

    S = {}
    for path in raw_paths:
        sid,deploy = get_sid(path),get_deploy(path)
        if sid in S:
            if deploy in S[sid]: S[sid][deploy] += [path]
            else:                S[sid][deploy]  = [path]
        else:                    S[sid]  = {deploy:[path]}
    ss = sorted(S)
    if args.sids is None: sids = ss
    for s in ss:
        if s not in sids: S.pop(s)
    print('%s sids were located among the image paths'%len(S))
    print('%s total deployments were found among the image paths'%(sum([len(S[sid]) for sid in S])))

    #[S] read exif and time sort with temporal trigger removal routine
    O = temporally_order_paths(S)
    for sid in O:
        for deploy in O[sid]:
            if len(O[sid][deploy]['trigs'])>0: #[0]remove triggered events
                trigs_dir,trig = out_dir+'/trigs',1
                if not os.path.exists(trigs_dir): os.mkdir(trigs_dir)
                print('copying %s triggered images to the trigs folder:%s'%(len(O[sid][deploy]['trigs']),trigs_dir))
                for r in O[sid][deploy]['trigs']:
                    img_name = trigs_dir+'/S%s_D%s_I%s.JPG'%(sid,deploy,trig)
                    cv2.imwrite(img_name,cv2.imread(r[-1]))
                    trig += 1
            if len(O[sid][deploy]['exif'])>0: #[0]remove triggered events
                exif_dir,exif = out_dir+'/exif',1
                if not os.path.exists(exif_dir): os.mkdir(exif_dir)
                print('copying %s images with non-valid dates to the exif folder:%s'%\
                      (len(O[sid][deploy]['exif']),exif_dir))
                for r in O[sid][deploy]['exif']:
                    img_name = exif_dir+'/S%s_D%s_I%s.JPG'%(sid,deploy,exif)
                    cv2.imwrite(img_name,cv2.imread(r[-1]))
                    exif += 1
    #partitioning ))))))))))))))))))))))))))))))
    H,P,T = [],{i:[] for i in range(cpus)},{}
    for sid in O:
        for deploy in O[sid]:
            H += [(len(O[sid][deploy]['order']),O[sid][deploy]['order'])]
    H = sorted(H,key=lambda x: x[0])[::-1]
    for i in range(len(H)): P[i%cpus] += [H[i]]
    for cpu in P:
        T[cpu] = {'n':0,'imgs':{}}
        for d in range(len(P[cpu])):
            if P[cpu][d][0]>0: #sids can be duplicates, deploys are unique here...
                sid,deploy = get_sid(P[cpu][d][1][0][-1]),get_deploy(P[cpu][d][1][0][-1])
                if sid not in T[cpu]['imgs']: T[cpu]['imgs'][sid] = {}
                T[cpu]['imgs'][sid][deploy] = P[cpu][d][1]
                T[cpu]['n'] += P[cpu][d][0]
    n_images = sum([T[cpu]['n'] for cpu in T])
    print('partitioned %s total images to %s processors'%(n_images,cpus))
    #partitioning ))))))))))))))))))))))))))))))

    start = time.time()
    p1 = mp.Pool(cpus)
    for cpu in T:  # balanced sid/deployments in ||
        print('dispatching %s images to core=%s'%(T[cpu]['n'],cpu))
        p1.apply_async(process_image_partitions,
                       args=(T[cpu]['imgs'],params),
                       callback=collect_results)
        time.sleep(0.1)
    p1.close()
    p1.join()
    stop  = time.time()
    print('processed %s images in %s sec using %s cpus'%(n_images,round(stop-start,2),cpus))
    print('or %s images per sec'%(n_images/(stop-start)))

    if args.json_img_map:
        print('preparing json image map...')
        generate_img_map_json(out_dir,out_dir+'/img_map.json')

    if args.label_sub_folders:
        I = {}
        for img_path in glob.glob(out_dir+'/passed*/*.JPG'):
            img_name = img_path.split('/')[-1]
            label = int(img_name.split('.JPG')[0].split('_L')[-1])
            if label in I: I[label] += [img_path]
            else:          I[label]  = [img_path]
        for l in I: I[l] = sorted(I[l])

        #print out the ratio of passed/total
        for l in sorted(I):
            in_labels = len(glob.glob(in_dir+'/label_%s/*.JPG'%l))
            print('label=%s:  %s/%s or %s passed the filtration process'%(l,len(I[l]),in_labels,round(len(I[l])/in_labels,2)))

        for l in sorted(I):
            img_dir = '/'.join(I[l][0].split('/')[:-1])+'/label_%s/'%l
            if not os.path.exists(img_dir): os.mkdir(img_dir)
            for i in range(len(I[l])):
                img_name = I[l][i].split('/')[-1]
                os.rename(I[l][i],img_dir+img_name)



