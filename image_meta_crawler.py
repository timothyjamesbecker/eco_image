import os
import time
import glob
import multiprocessing as mp
import argparse
import utils

des="""image meta data crawler"""
parser = argparse.ArgumentParser(description=des,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-i', '--in_dir',type=str, help='image input directory')
parser.add_argument('-o', '--out_dir',type=str,help='output_directory')
parser.add_argument('-p', '--cpus',type=int,help='number of processors')
args = parser.parse_args()

if args.in_dir is not None and os.path.exists(args.in_dir):
    in_dir = args.in_dir
else: raise IOError

if args.out_dir is not None:
    out_dir = args.out_dir
    if not os.path.exists(out_dir): os.mkdir(out_dir)
else: raise IOError

if args.cpus is not None:
    cpus = args.cpus
else: cpus = 1

result_list = [] #async queue to put results for || stages
def collect_results(result):
    result_list.append(result)

def map_cameras_to_sid(sid_images,sid,tag_set):
    S = {sid:{}}
    for path in sid_images:
        T = utils.get_exif_tags(path,tag_set)
        v = tuple(T[t] for t in sorted(T.keys()))
        if S[sid].has_key(v): S[sid][v] += 1
        else:                 S[sid][v]  = 1
    return S

if __name__ == '__main__':
    start=time.time()
    #initialize----------------------------------------------------------------------------------------------------
    #file naming pattern: sid_segmentname_startdate_stop_date (x).JPG
    #where a sid is a unique latitude/longitude pair (pos int) and a segment name can have multiple sids associated
    sids=sorted(list(set([int(x.rsplit('_')[0].rsplit('/')[-1]) for x in glob.glob(in_dir+'/*_*_*_*')])))
    print(sids)
    tag_set=set(['Image Make','Image Model'])
    S={}  #map every camera make and model to the sids
    for sid in sids: S[sid]=glob.glob(in_dir+'/%s_*'%sid)
    #map/scatter---------------------------------------------------------------------------------------------------
    p1=mp.Pool(processes=cpus)
    for sid in sids: # each site in ||
        sid_images = glob.glob(in_dir+'/%s_*'%sid)
        print('processing %s images for sid=%s'%(len(sid_images),sid))
        p1.apply_async(map_cameras_to_sid,
                       args=(sid_images,sid,tag_set),
                       callback=collect_results)
        time.sleep(0.1)
    p1.close()
    p1.join()
    #reduce/gather-------------------------------------------------------------------------------------------------
    L={}
    for result in result_list:
        L[result.keys()[0]] = result[result.keys()[0]]
    stop=time.time()
    #share/print/write analysis------------------------------------------------------------------------------------
