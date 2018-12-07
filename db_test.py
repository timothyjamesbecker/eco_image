#!/usr/bin/env python
import os
import time
import argparse
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mysql_connector as mysql #uses mysql.connector 8.13+, and optionally parimiko->sshtunnel

des="""db CRUD tester and data image sampler\nTimothy James Becker"""
parser = argparse.ArgumentParser(description=des,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--host',type=str, help='host name\t\t\t[None]')
parser.add_argument('--port',type=int,help='port number\t\t\t[None]')
parser.add_argument('--db',type=str,help='db schema name\t\t\t[None]')
parser.add_argument('--table',type=str,help='table name\t\t\t[None]')
parser.add_argument('--blob',action='store_true',help='test image blob insertion\t[None]')
parser.add_argument('--tunnel',action='store_true',help='use a ssh tunnel\t\t[False]')
parser.add_argument('--select_query',type=str,help='mysql raw select query\t\t[None]')
parser.add_argument('--verbose',action='store_true',help='print/plot everything\t\t[False]')
args = parser.parse_args()

if not args.host is None:
    host = args.host
else:
    host = '127.0.0.1'
if not args.port is None:
    port = args.port
else:
    port = 3306
if not args.db is None:
    db = args.db
else:
    raise AttributeError
if not args.table is None:
    tbl = args.table
else:
    tbl = 'test'

uid,pwd=False,False
local_path = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(local_path+'/db.cfg'):
    with open(local_path+'/db.cfg','r') as f:
        raw = f.readlines()
        uid = raw[0].replace('\n','')
        pwd = raw[1].replace('\n','')
ssh_uid,ssh_pwd = False,False
if os.path.exists(local_path+'/tunnel.cfg'):
    with open(local_path+'/tunnel.cfg','r') as f:
        raw = f.readlines()
        ssh_uid = raw[0].replace('\n','')
        ssh_pwd = raw[1].replace('\n','')

#random lat,lon generator with 6 decimal places
def get_pos():
    return round(random.randint(-180,180)+random.random(),6)

if __name__ == '__main__':
    start = time.time()
    #manual connector testing v is an injection safe value list, ? are the positional holders for values
    SQ = [{'sql':'drop table if exists %s.%s;'%(db,tbl)},
          {'sql':'create table %s.%s(pk int primary key, des text, lat float(10), lon float(10), data mediumblob);'%(db,tbl)},
          {'sql':'insert into %s.%s value (?,?,?,?,?),(?,?,?,?,?);'%(db,tbl),
           'v':[random.randint(0,10),'Hello',get_pos(),get_pos(),None,
                random.randint(10,20),'Goodbye',get_pos(),get_pos(),None]},
          {'sql':'select * from %s.%s'%(db,tbl)}]
    RQ,SD,RD = [],[],[]
    if args.select_query is not None:
        SD += [{'sql':args.select_query}]
    if args.verbose: print(SQ)
    if args.verbose: print(SD)
    with mysql.MYSQL(host=host,port=port,db=db,uid=uid,pwd=pwd,
                     ssh_uid=ssh_uid,ssh_pwd=ssh_pwd,
                     delim='?',ssh_tunnel=args.tunnel) as dbo:
        dbo.set_SQL_V(SQ)
        RQ = dbo.run_SQL_V()
    if args.verbose: print(RQ)
    if SD != []:
        with mysql.MYSQL(host=host,port=port,db=db,uid=uid,pwd=pwd,
                         ssh_uid=ssh_uid,ssh_pwd=ssh_pwd,
                         delim='?',ssh_tunnel=args.tunnel) as dbo:
            dbo.set_SQL_V(SD)
            RD = dbo.run_SQL_V()
        if args.verbose: print(RD)

    if args.blob:
        #generate a binary in memory str and insert into mediumblob field
        local_path = os.path.dirname(os.path.abspath(__file__))
        img_path = local_path+'/data/camera_examples/M-999i/15100_SpruceSwampCreek_051018_062018 (1).JPG'
        with open(img_path,'rb') as f: s = f.read() #str type
        in_img = cv2.imdecode(np.frombuffer(bytearray(s),dtype=np.uint8),-1)
        SQ = [{'sql':'insert into %s.%s value (?,?,?,?,?);'%(db,tbl),'v':[30,'Hello',get_pos(),get_pos(),s]},
              {'sql':'select * from %s.%s where pk = ?;'%(db,tbl),'v':[30]}]
        with mysql.MYSQL(host=host,port=port,db=db,uid=uid,pwd=pwd,
                         ssh_uid=ssh_uid,ssh_pwd=ssh_pwd,
                         delim='?',ssh_tunnel=args.tunnel) as dbo:
            dbo.set_SQL_V(SQ)
            RQ = dbo.run_SQL_V()

        #check the image by converting from binary in memory to plot
        out_img = cv2.imdecode(np.frombuffer(RQ[0][0]['data'],dtype=np.uint8),-1)
        print('bit for bit storage ? %s'%(np.array_equal(in_img,out_img)))
        if args.verbose:
            plt.imshow(cv2.cvtColor(out_img,cv2.COLOR_BGR2RGB))
            plt.show()

    stop = time.time()
    print('db tests completed in %s sec'%round(stop-start,2))
