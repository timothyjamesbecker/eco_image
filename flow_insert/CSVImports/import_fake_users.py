import mysql_connector as con
import time
import csv

# read cfg for host and database to connect to
with open("db.cfg") as f:
    host = f.readline().rstrip('\n')
    database = f.readline().rstrip('\n')

# read cfg for credentials (username and password to DB)
    # TODO: decrypt using a key found in another file
with open("cred.cfg") as f:
    usr = f.readline().rstrip('\n')
    pwd = f.readline().rstrip('\n')

# connect
db = con.MYSQL(host, database, usr, pwd)

# time query
start = time.clock()

# variable to be injected
id = []
# query that injects ID into %s
query_base = "INSERT INTO user (UserID, UserName) VALUES "
query = query_base

with open('C:\\Users\\Nick\\Desktop\\Working\\Flow_Insert\\Data\\FakeUsers.csv', mode='r') as infile:
    reader = csv.reader(infile)
    infile.readline()

    values = list(reader)
    buffer_size = 2
    i = 0
    for row in values:
        i = i + 1
        if i >= buffer_size:
            print('\r\ni >= buffer_size')
            query = query[:-1] + ";"
            print('query = ')
            print(query)
            print("values = ")
            print(id)
            r = db.query(query, id, False)
            #print('query results below:')
            #print(r)
            i = 0
            id = []
            query = query_base
        row[0] = None
        print('row = %s', row)
        query += '(%s, %s),'
        id.extend(row)
    if i != 0:
        print('\r\noutside of loop:')
        query = query[:-1] + ";"
        #print('query results below:')
        print('query = ')
        print(query)
        print("values = ")
        print(id)
        r = db.query(query, id, False)
        #print(r)

"""r = db.query("select * from user where UserID=%s;", [(id)], True)"""

# stop timing
stop = time.clock()
