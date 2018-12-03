#!/usr/bin/env python
import os
import time
import argparse
import random
import mysql_connector as mysql #uses mysql.connector 8.13+
import mysql.connector as msc

des="""db CRUD tester and data sampler\nTimothy James Becker"""
parser = argparse.ArgumentParser(description=des,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--host',type=str, help='host name\t[None]')
parser.add_argument('--port',type=int,help='port number\t[None]')
parser.add_argument('--db',type=str,help='db schema name\t[None]')
parser.add_argument('--table',type=str,help='table name\t[None]')
parser.add_argument('--select_query',type=str,help='mysql raw select query\t[None]')
parser.add_argument('--wrapper',action='store_true',help='use the mysql_connector.MYSQL class\t[False]')
parser.add_argument('--verbose',action='store_true',help='print everything to stdout\t[False]')
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
if os.path.exists(local_path+'/test.cfg'):
    with open(local_path+'/test.cfg','r') as f:
        raw = f.readlines()
        uid = raw[0].replace('\n','')
        pwd = raw[1].replace('\n','')

#random lat,lon generator with 6 decimal places
def get_pos():
    return round(random.randint(-180,180)+random.random(),6)

if __name__ == '__main__':
    #manual connector testing v is an injection safe value list, ? are the positional holders for values
    QS = [{'sql':'drop table if exists %s.%s;'%(db,tbl)},
          {'sql':'create table %s.%s(pk int primary key, des text, lat float(10), lon float(10));'%(db,tbl)},
          {'sql':'insert into %s.%s value (?,?,?,?),(?,?,?,?);'%(db,tbl),
           'v':[random.randint(0,10),'Hello',get_pos(),get_pos(),random.randint(10,20),'Goodbye',get_pos(),get_pos()]},
          {'sql':'select * from %s.%s'%(db,tbl)}]
    SD,RD = [],[]
    if args.select_query is not None:
        SD += [{'sql':args.select_query}]
    if args.verbose: print(QS)
    if args.verbose: print(SD)
    if args.wrapper:

        with mysql.MYSQL(host=host,port=port,db=db,uid=uid,pwd=pwd,delim='?') as dbo:
            dbo.set_SQL_V(QS)
            res = dbo.run_SQL_V()
            print(res)

        with mysql.MYSQL(host=host,port=port,db=db,uid=uid,pwd=pwd,delim='?') as dbo:
            dbo.set_SQL_V(SD)
            RD += [dbo.run_SQL_V()]
            print(RD)
    else:

        res = []
        conn = msc.connect(host=host,database=db,port=port,user=uid,password=pwd)
        conn.set_charset_collation('utf8','utf8_general_ci')
        for i in range(len(QS)):
            sql,v = QS[i]['sql'].replace('?','%s'),()
            if QS[i].has_key('v'): v = tuple(QS[i]['v'])
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql,v)
            if cursor.with_rows: res += [cursor.fetchall()]
            cursor.close()
        conn.commit()
        print(res)

        conn = msc.connect(host=host,database=db,port=port,user=uid,password=pwd)
        conn.set_charset_collation('utf8','utf8_general_ci')
        for i in range(len(SD)):
            sql,v = SD[i]['sql'].replace('?','%s'),()
            if SD[i].has_key('v'): v = tuple(SD[i]['v'])
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql,v)
            if cursor.with_rows: RD += [cursor.fetchall()]
            cursor.close()
        conn.commit()
        print(RD)

