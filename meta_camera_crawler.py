#!/usr/bin/env python
import os
import argparse
import copy
import time
import glob
import multiprocessing as mp
import piexif
import utils

des="""image meta camera id crawler"""
parser = argparse.ArgumentParser(description=des,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-i', '--map_file_path',type=str, help='image file path to camera id path')
parser.add_argument('-p', '--cpus',type=int,help='number of processors')
args = parser.parse_args()

if args.map_file_path is not None and os.path.exists(args.map_file_path):
    map_file_path = args.map_file_path
else: raise IOError

if args.cpus is not None:
    cpus = args.cpus
else: cpus = 1

#result data store for shared memory
result_list = [] #async queue to put results for || stages
def collect_results(result):
    result_list.append(result)

#worker unit-----------------------------
def set_image_meta(M,p):
    L = {}
    print('cpu %s has %s images'%(p,len(M)))
    for f in M:
        try:
            E = piexif.load(os.path.abspath(f))   #read exif bytes from the image
            E['0th'][270] = M[f]  #set the tag code to the value in the map M[f]
            piexif.insert(piexif.dump(E),os.path.abspath(f))   #write the new exif bytes into the image
            L += [os.path.abspath(f)]             #save the result filename to report back
        except Exception as e:
            pass
    return L

#entry point------------
if __name__=='__main__':
    start=time.time()
    #load the file to camera map and check the paths------------------------------------------------------
    with open(map_file_path,'r') as f:
        C={l.replace('\n','').rsplit(',')[0]:l.replace('\n','').rsplit(',')[1] for l in f.readlines()[2:]}
    #check to see if the path is correct
    N = []
    for c in C:
        if not os.path.exists(c): N += [c]
    for n in N: C.pop(n)
    print('%s files have been found, proceeding to set exif metadata'%len(C))
    #partition the maps across cpus....-------------------------------------
    P,cs = {i:{} for i in range(cpus)},C.keys()
    for i in range(len(C)): P[i%cpus][cs[i]] = C[cs[i]]
    #dispatch the processes to all cores-------------------------------------
    p1=mp.Pool(processes=cpus)
    for p in P:  # each site in ||
        print('dispatching %s images to cpu %s'%(len(P[p]),p))
        p1.apply_async(set_image_meta,
                       args=(P[p],p),
                       callback=collect_results)
        time.sleep(0.1)
    p1.close()
    p1.join()
    #collect results---------------------------------------------------------
    X = []
    for l in result_list: X += l #append the processed files into the master list
    stop = time.time()
    print('processed %s image files in %s sec'%(len(X),round(stop-start,2)))
#exit point----------------