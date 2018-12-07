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

des="""bottom-center priority image crop/resize crawler\nTimothy James Becker"""
parser = argparse.ArgumentParser(description=des,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--host',type=str, help='host name\t\t\t[None]')
parser.add_argument('--port',type=int,help='port number\t\t\t[None]')
parser.add_argument('--db',type=str,help='db schema name\t\t\t[None]')
parser.add_argument('--img_table',type=str,help='image table name\t\t\t[None]')
parser.add_argument('--inv_table',type=str,help='inventory table name\t\t\t[None]')
parser.add_argument('--tunnel',action='store_true',help='use a ssh tunnel\t\t[False]')
parser.add_argument('--img_batch',type=int,help='number of images to proccess\t\t\t[1]')
parser.add_argument('--img_offset',type=int,help='number of records to offset the start of the select,insert\t[0]')
parser.add_argument('--cpus',type=int,help='number of cores to use to process the images\t[1]')
parser.add_argument('--verbose',action='store_true',help='print/plot everything\t\t[False]')
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
    R = []
    # for i in range(params['limit'): #one at a time, then b at a time
    #get the PK and the RawData value
    S = [{'sql':'select * from %s.%s limit %s offset %s;'%(db,tbl,params['limit'],0+params['offset'])}]
    print(S)
    with mysql.MYSQL(host=host,port=port,db=db,uid=uid,pwd=pwd,
                     ssh_uid=ssh_uid,ssh_pwd=ssh_pwd,
                     delim='?',ssh_tunnel=args.tunnel) as dbo:
        dbo.set_SQL_V(S)
        R = dbo.run_SQL_V()
    try:
        row = R[0][0]
        c_id,t_id = row['CameraNumber'],row['TimeTaken']       #pk = (c_id,t_id)
        raw_img   = utils.blob2img(row['RawData'])             #never touch original
        seg_mult = utils.get_camera_seg_mult(camera[c_id])     #get camera model
        seg      = utils.get_seg_line(raw_img,mult=seg_mult)   #get dominant seg line => footer
        raw_img  = utils.crop_seg(raw_img,seg)                 #slice image
        web_img  = utils.resize(raw_img,
                                width=params['res']['web'][0],
                                height=params['res']['web'][1])
        web_blob = utils.img2blob(web_img)
        thb_img  = utils.resize(raw_img,
                                width=params['res']['thb'][0],
                                height=params['res']['thb'][1])
        thb_blob = utils.img2blob(thb_img)
        #write it back pk => camera and time
        U = [{'sql':'update %s.%s set WebData = ?,ThumbData = ? where %s = ? and %s = ?;'%\
                  (db,tbl,'CameraNumber','TimeTaken'),
              'v':(web_blob,thb_blob,c_id,t_id)}]
        with mysql.MYSQL(host=host,port=port,db=db,uid=uid,pwd=pwd,
                         ssh_uid=ssh_uid,ssh_pwd=ssh_pwd,
                         delim='?',ssh_tunnel=args.tunnel) as dbo:
            dbo.set_SQL_V(U)
            R = dbo.run_SQL_V()
            success = True
    except Exception as E:
        success = False
        pass
    return [success]

if __name__ == '__main__':
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
    print(SQL)
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
        w = n/cpus
        f = n%cpus
        P = {c:w for c in range(cpus)}
        P[0] += f
        ks = P.keys()
        for p in ks: #clear any zero limit chucks
            if P[p] == 0: P.pop(p) #when cpus are set higher than rows
    offset,params,res = 0,{},{'web':(1024,576),'thb':(40,40)} #16:9 aspect, 4:3 aspect
    for p in P:
        params[p] = {'offset':offset,'limit':P[p],'res':res}
        offset += P[p]
    print(params)

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
    for l in result_list: X += l

    S=[{'sql':'select * from %s.%s limit %s offset %s;'%(db,img_tbl,1,0)}]
    print(S)
    with mysql.MYSQL(host=host,port=port,db=db,uid=uid,pwd=pwd,
                     ssh_uid=ssh_uid,ssh_pwd=ssh_pwd,
                     delim='?',ssh_tunnel=args.tunnel) as dbo:
        dbo.set_SQL_V(S)
        R = dbo.run_SQL_V()
    if(len(R)>0 and len(R[0][0])>0):
        utils.plot(utils.blob2img(R[0][0]['RawData']))
        utils.plot(utils.blob2img(R[0][0]['WebData']))
        utils.plot(utils.blob2img(R[0][0]['ThumbData']))

