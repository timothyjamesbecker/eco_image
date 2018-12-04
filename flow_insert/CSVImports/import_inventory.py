# done, works
import mysql_connector as con
from datetime import datetime
import time
import csv

# read cfg for host and database to connect to
with open("../db.cfg") as f:
    host = f.readline().rstrip('\n')
    database = f.readline().rstrip('\n')

# read cfg for credentials (username and password to DB)
    # TODO: decrypt using a key found in another file
with open("../cred.cfg") as f:
    usr = f.readline().rstrip('\n')
    pwd = f.readline().rstrip('\n')

# connect
db = con.MYSQL(host, database, usr, pwd)

# time query
start = time.clock()

# variable to be injected
id = []
# query that injects ID into %s
query_base = """insert into inventory (CameraModel,
 SerialNumber, CameraNumber, DateReceived,
 LastLocation, StillInUse, Comments)
 values """

query = ''
with open('../Data/InventoryDataClean.csv', mode='r') as infile:
    reader = csv.reader(infile)
    infile.readline()

    values = list(reader)
    buffer_size = 200
    i = 0
    for row in values:
        i = i + 1
        # the indices of the columns whose types/formats need to be changed
        dates = [3]
        ints = [2, 5]
        chars = [5]
        # replace null values with None and change types
        for j in range(0, len(row)):
            if row[j].rstrip(' ') == '':
                row[j] = None
            elif j in dates:
                d = datetime.strptime(row[j], "%m/%d/%Y")
                ts = time.mktime(d.timetuple())
                row[j] = int(ts)
            elif j in ints:
                row[j] = int(row[j])
            elif j in chars:
                row[j] = str(row[j])
        query += '(' + '%s,'*(len(row)-1) + '%s),'
        id.extend(row)
        if i >= buffer_size:
            query = query_base + query[:-1] + ";"
            r = db.query(query, id)
            i = 0
            id = []
            query = ''
    if i != 0:
        query = query_base + query[:-1] + ";"
        r = db.query(query, id)
        query = ''

"""r = db.query("select * from user where UserID=%s;", [(id)], True)"""

# stop timing
stop = time.clock()
