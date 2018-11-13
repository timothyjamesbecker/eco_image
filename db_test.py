#!/usr/bin/env python
import os
import time
import argparse
import mysql_connector as msc

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
if os.path.exists(local_path+'/flow.cfg'):
    with open(local_path+'/flow.cfg','r') as f:
        raw = f.readlines()
        uid = raw[0].replace('\n','')
        pwd = raw[1].replace('\n','')

#wrapper library testing--------------------------------------------------------------------------------------------
with msc.MSQL(host=host,port=port,db=db,uid=uid,pwd=pwd) as dbo:
    C = """drop table if exists %s;
           create table %s(pk int primary key not null, description text, lat float(10), lon float(10));"""%(tbl,tbl)
    R = """select * from %s;"""%tbl
    U =  """insert into %s values(13,"Needed",41.234526,-71.425362);
            insert into %s values(21,"To Test",41.273652,72.162534);"""%(tbl,tbl)

    D =  """truncate table %s;"""%tbl
    Q = [C,R,U,D]
    for q in Q:
        res = dbo.query(q,[],r=True)
        print(res)
#--------------------------------------------------------------------------------------------------------------------

#manual connector testing
# conn = msc.connect(host=unicode(host),port=unicode(port),database=unicode(db),user=unicode(uid),password=unicode(pwd))
# cursor = conn.cursor(dictionary=True)
# cursor.execute(sql,v)
# # for row in cursor: res.append(row)
# res = cursor.fetchall()
# cursor.close()
# conn.close()
# print(res)