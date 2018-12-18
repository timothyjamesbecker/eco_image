#!/usr/bin/env python
import os
import time
import argparse
import random
import numpy as np
import cv2
import multiprocessing as mp
import mysql_connector as mysql #uses mysql.connector 8.13+, and optionally parimiko->sshtunnel
import utils

des="""
---------------------------------------------------
Bottom-Center Priority Image Crop/Resize Crawler
Timothy James Becker 12-18-2018
---------------------------------------------------"""
parser = argparse.ArgumentParser(description=des.lstrip(" "),formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--host',type=str, help='host name\t\t\t\t\t\t\t[None]')
parser.add_argument('--port',type=int,help='port number\t\t\t\t\t\t\t[None]')
parser.add_argument('--db',type=str,help='db schema name\t\t\t\t\t\t\t[None]')
parser.add_argument('--img_table',type=str,help='image table name\t\t\t\t\t\t[None]')
parser.add_argument('--inv_table',type=str,help='inventory table name\t\t\t\t\t\t[None]')
parser.add_argument('--tunnel',action='store_true',help='use a ssh tunnel\t\t\t\t\t\t[False]')
parser.add_argument('--img_batch',type=int,help='number of images to proccess\t\t\t\t\t[1]')
parser.add_argument('--img_q',type=int,help='number of images to retrieve per query\t\t\t\t[1]')
parser.add_argument('--img_offset',type=int,help='number of records to offset the start of the select,insert\t[0]')
parser.add_argument('--cpus',type=int,help='number of cores to use to process the images\t\t\t[1]')
parser.add_argument('--verbose',action='store_true',help='print/plot everything\t\t\t\t\t\t[False]')
args = parser.parse_args()

if args.host is not None:
    host = args.host
else:
    host = '127.0.0.1'
if args.port is not None:
    port = args.port
else:
    port = 3306
if args.db is not None:
    db = args.db
else:
    db = 'flow_observation'
if args.img_table is not None:
    img_tbl = args.img_table
else:
    img_tbl = 'jpgImage'
if args.inv_table is not None:
    inv_tbl = args.inv_table
else:
    inv_tbl = 'inventory'
if args.img_batch is not None:
    img_batch = args.img_batch
else:
    img_batch = 1
if args.img_q is not None:
    img_q = args.img_q
else:
    img_q = 1
if args.img_offset is not None:
    img_offset = args.img_offset
else:
    img_offset = 0
if args.cpus is not None:
    cpus = args.cpus
else:
    cpus = 4

result_list = [] #async queue to put results for || stages
def collect_results(result):
    result_list.append(result)

def crop_resize_images(db,tbl,camera,params):
    offset,q_lim,t_lim = params['offset'],params['q_lim'],params['t_lim']
    w_res,t_res = params['res']['web'],params['res']['thb']
    R,T = {},{}
    for i in range(0,t_lim,q_lim):
        if (i+q_lim) <= t_lim: limit = q_lim
        else:                  limit = i+q_lim-t_lim
        S = [{'sql':'select * from %s.%s limit %s offset %s;'%(db,tbl,limit,offset+i)}]
        with mysql.MYSQL(host=host,port=port,db=db,uid=uid,pwd=pwd,
                         ssh_uid=ssh_uid,ssh_pwd=ssh_pwd,
                         delim='?',ssh_tunnel=args.tunnel) as dbo:
            dbo.set_SQL_V(S)
            R = dbo.run_SQL_V()
        #iterate on the results
        if len(R)>0:
            j = 0
            for row in R[0]:
                if len(row)>0:
                    try:
                        c_id,t_id = row['CameraNumber'],row['TimeTaken']        #pk = (c_id,t_id)
                        raw_img   = utils.blob2img(row['RawData'])              #never touch original
                        seg_mult  = utils.get_camera_seg_mult(camera[c_id])     #get camera model
                        seg       = utils.get_seg_line(raw_img,mult=seg_mult)   #get dominant seg line => footer
                        raw_img   = utils.crop_seg(raw_img,seg)                 #slice image
                        web_img   = utils.resize(raw_img,w_res[0],w_res[1])
                        web_blob  = utils.img2blob(web_img)
                        thb_img   = utils.resize(raw_img,t_res[0],t_res[1])
                        thb_blob  = utils.img2blob(thb_img)
                        #write it back pk => camera and time
                        U = [{'sql':'update %s.%s set WebData = ?,ThumbData = ? where %s = ? and %s = ?;'%\
                                    (db,tbl,'CameraNumber','TimeTaken'),
                              'v':(web_blob,thb_blob,c_id,t_id)}]
                        with mysql.MYSQL(host=host,port=port,db=db,uid=uid,pwd=pwd,
                                         ssh_uid=ssh_uid,ssh_pwd=ssh_pwd,
                                         delim='?',ssh_tunnel=args.tunnel) as dbo:
                            dbo.set_SQL_V(U)
                            R = dbo.run_SQL_V()
                            T[offset+i+j] = True
                    except Exception as E:
                        T[offset+i+j] = [E.message]
                        pass
                j += 1
    return T

if __name__ == '__main__':
    start = time.time()
    uid,pwd = False,False
    local_path = utils.local_path()
    if os.path.exists(local_path+'/db.cfg'):
        with open(local_path+'/db.cfg','r') as f:
            raw = f.readlines()
            uid = raw[0].replace('\n','')
            pwd = raw[1].replace('\n','')
    ssh_uid,ssh_pwd=False,False
    if os.path.exists(local_path+'/tunnel.cfg'):
        with open(local_path+'/tunnel.cfg','r') as f:
            raw = f.readlines()
            ssh_uid = raw[0].replace('\n','')
            ssh_pwd = raw[1].replace('\n','')

    start = time.time()
    #(1) get the total count of the table for partitioning
    SQL = [{'sql':'select count(*) as n from %s.%s;'%(db,img_tbl)},
           {'sql':'select * from %s.%s;'%(db,inv_tbl)}]
    R,camera,n = [],{},0
    if args.verbose: print(SQL)
    with mysql.MYSQL(host=host,port=port,db=db,uid=uid,pwd=pwd,
                     ssh_uid=ssh_uid,ssh_pwd=ssh_pwd,
                     delim='?',ssh_tunnel=args.tunnel) as dbo:
        dbo.set_SQL_V(SQL)
        R = dbo.run_SQL_V()
        try:
            n = R[0][0]['n']
            for row in R[1]:
                camera[row['CameraNumber']] = row['CameraModel']
        except Exception as E:
            print(E.err)
            pass
    if n > 0:
        n = min(n,img_batch)
        if args.tunnel:
            P = {0:n} #can only bind one port
        else:
            w = n/cpus
            f = n%cpus
            P = {c:w for c in range(cpus)}
            P[0] += f
            ks = P.keys()
            for p in ks: #clear any zero limit chucks
                if P[p] == 0: P.pop(p) #when cpus are set higher than rows
    offset,params,res = img_offset,{},{'web':(1024,576),'thb':(256,192)} #16:9 aspect, 4:3 aspect
    for p in P: #t_lim is the total images per cpu, q_lim is the number of images per query
        params[p] = {'offset':offset,'t_lim':P[p],'q_lim':img_q,'res':res}
        offset += P[p]
    print('cameras: %s'%camera)
    print('params: %s'%params)

    p1 = mp.Pool(processes = cpus)
    for p in params:  # each site in ||
        p1.apply_async(crop_resize_images,
                       args=(db,img_tbl,camera,params[p]),
                       callback=collect_results)
        time.sleep(0.1)
    p1.close()
    p1.join()
    #collect results---------------------------------------------------------
    X = []
    for l in result_list: X += [l]
    if args.verbose: print('results: %s'%X)
    stop = time.time()
    print('finished processing %s rows in %s sec'%(n,round(stop-start,2)))
    if args.verbose:
        #print out two of the last ones effected to test functionality------------------------------------
        S = [{'sql':'select * from %s.%s limit %s offset %s;'%(db,img_tbl,2,(n-2)+img_offset)}]
        print(S)
        with mysql.MYSQL(host=host,port=port,db=db,uid=uid,pwd=pwd,
                         ssh_uid=ssh_uid,ssh_pwd=ssh_pwd,
                         delim='?',ssh_tunnel=args.tunnel) as dbo:
            dbo.set_SQL_V(S)
            R = dbo.run_SQL_V()
        if len(R)>0:
            for row in R[0]:
                if len(row)>0:
                    utils.plot(utils.blob2img(row['RawData']))
                    if row['WebData'] is not None:
                        utils.plot(utils.blob2img(row['WebData']))
                    if row['ThumbData'] is not None:
                        utils.plot(utils.blob2img(row['ThumbData']))

