#!/usr/bin/env python
import os
import time
import argparse
import mysql_connector

des="""eco image db connection and CRUD tester"""
parser = argparse.ArgumentParser(description=des,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--host',type=str, help='host name\t[None]')
parser.add_argument('--port',type=int,help='port number\t[None]')
parser.add_argument('--db',type=str,help='db schema name\t[None]')
parser.add_argument('--table',type=str,help='table name\t[None]')
args = parser.parse_args()

if not args.host is None:
    host = args.host
else:
    raise AttributeError
if not args.port is None:
    port = args.port
else:
    raise AttributeError
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
if os.path.exist(local_path+'/flow.cfg'):
    with open(local_path+'/flow.cfg','r') as f:
        raw = f.readlines()
        uid = raw[0].replace('\n','')
        pwd = raw[1].replace('\n','')
with mysql_connector.MYSQL(host=host,port=port,db=db,uid=uid,pwd=pwd) as dbo:
    SQL = 'select * from %s.%s;'%(db,tbl)
    res = dbo.query(SQL,[],r=True)
    print(res)