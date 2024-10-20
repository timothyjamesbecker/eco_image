import os
import glob
import cv2
import time
import json
import piexif
import datetime as dt
import numpy as np
import argparse
import utils

def get_sid(path):
    ss,sid = path.split('/'),-1
    if path.split('/')[-1].split('_')[0].isdigit():
        sid = int(path.split('/')[-1].split('_')[0])
    return sid

def fix_file_names(path):
    ss   = path.split('/')
    stub = ss[-2]
    if ss[-1].find(stub)>-1: #file name has the folder name in it
        if stub.split('_')[-1].isdigit(): #is the folder name ending with a number
            new_path = stub+'_'+ss[-1].replace(stub,'')
        else:                             #it is ending with an initial
            new_path = stub+'_'+ss[-1].replace(stub + '_', '')
    else:                    #file name doesn't have the folder name in it
        if stub.split('_')[-1].isdigit():
            new_path = ss[-2]+'_'+ss[-1]
        else:
            new_path = '_'.join(ss[-2].split('_')[:-1])+'_'+ss[-1]
    new_path = '/'.join(ss[:-1])+'/'+new_path
    if path!=new_path: os.rename(path,new_path)
    return [path,new_path]

def get_label(path):
    label = 0
    if path.find('label_')>-1:
        label = int(path.split('label_')[-1].split('/')[0])
    elif path.endswith('_L0.JPG') or path.endswith('_L1.JPG') or path.endswith('_L2.JPG') or\
            path.endswith('_L3.JPG') or path.endswith('_L4.JPG') or path.endswith('_L5.JPG') or path.endswith('_L6.JPG'):
        label = int(path.split('/')[-1].split('_L')[-1].split('.JPG')[0])
    else:
        if os.path.exists(path):
            try:
                E = piexif.load(path)
                if '0th' in E:
                    t =''
                    if 270 in E['0th']:
                        t = E['0th'][270].decode('ascii')
                    if t.isdigit(): label = int(t)
            except Exception as e:
                print(e,path)
    return label

def get_deploy(path):
    return '_'.join(path.split('/')[-2].split('_')[2:5]).split(' ')[0]

#read all images exif data to order and partition triggered events,returns: order,error,trigs
def temporally_order_paths(S,time_bin=5):
    O = {}
    for sid in S:
        O[sid] = {}
        for deploy in sorted(S[sid]):
            try:
                O[sid][deploy] = {'order':[],'exif':[],'trigs':[]}
                start = dt.datetime.strptime(deploy.split('_')[0],'%m%d%y')-dt.timedelta(seconds=1)
                stop  = dt.datetime.strptime(deploy.split('_')[1],'%m%d%y')+dt.timedelta(hours=24)
                #---------------------------------------------------------------------------------
                raw = []
                for i in range(len(S[sid][deploy])):
                    path  = S[sid][deploy][i]
                    label = get_label(path)
                    exif  = utils.read_exif_tags(path)
                    if 'Image Make' in exif: ct = exif['Image Make']
                    else:                    ct = 'Unknown'
                    if 'Image DateTime' in exif: ts = dt.datetime.strptime(exif['Image DateTime'],'%Y:%m:%d %H:%M:%S')
                    else:                        ts = dt.datetime.strptime('2020:01:01 00:00:00','%Y:%m:%d %H:%M:%S')
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
            except Exception as e:
                print(e)
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
    else: enh_hrs = 2
    if args.agg_hrs is not None:
        agg_hrs = args.agg_hrs
    else: agg_hrs = 2
    if args.equalize:
        equalize = True
    else: equalize = False
    if args.advanced:
        advanced = True
    else: advanced = False
    if args.winsize is not None:
        winsize = args.winsize
    else: winsize = 2
    if args.cpus is not None:
        cpus = args.cpus
    else: cpus = 1

    #sids = [19022]#,14434,14523,15244]

    params = {'width':width,'height':height,'enh_hrs':enh_hrs,'agg_hrs':agg_hrs,'equalize':equalize,'advanced':advanced,
              'mean':mean,'sharp':sharp,'winsize':winsize,'pad':0.2,'d_min':2.0,'pixel_dist':4*height//10,
              'write_labels':args.label_sub_folders}
    print('using params:%s'%params)
    raw_path = in_dir+'/*/*.JPG'

    ffids = [fix_file_names(path) for path in glob.glob(raw_path)]
    raw_paths = sorted(glob.glob(raw_path))
    if len(raw_paths)<0:
        print('raw image data is not ordered by label, proceeding to locate unlabeled JPGs...')
        raw_paths = sorted(glob.glob(in_dir+'/*.JPG'))
        print('located %s unlabeled image paths'%(len(raw_paths)))
    else: print('located %s label ordered image paths'%len(raw_paths))

    while out_dir[-1]=='/': out_dir = out_dir[:-1]
    out_dir = out_dir+'_ehrs%s_ahrs%s_%s/'%(params['enh_hrs'],params['agg_hrs'],('advanced' if params['advanced'] else 'basic'))
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    print('output directory has been created at:%s'%out_dir)
    params['out_dir'] = out_dir

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
    O = temporally_order_paths(S) #this will call get_lable and look for exif data...
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
    utils.process_image_partitions(T,params,cpus=cpus)
    #utils.worker_image_partitions(T[cpu]['imgs'],params)
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
            label = get_label(img_path)
            if label in I: I[label] += [img_path]
            else:          I[label]  = [img_path]
        for l in I: I[l] = sorted(I[l])

        for l in sorted(I):
            img_dir = '/'.join(I[l][0].split('/')[:-1])+'/label_%s/'%l
            if not os.path.exists(img_dir): os.mkdir(img_dir)
            for i in range(len(I[l])):
                img_name = I[l][i].split('/')[-1]
                os.rename(I[l][i],img_dir+img_name)
