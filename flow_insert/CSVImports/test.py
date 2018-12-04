import mysql_connector as con
import time

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
id = '1'
# query that injects ID into %s
s = "insert into user values (%s,%s),(%s,%s);"
v = [None, 'test', None, 'test1']
r = db.query(s, v)

# stop timing
stop = time.clock()
# print the result and rounded query time
print(f'\n{r}\ncompleted query in {round(stop-start, 6)} sec')
