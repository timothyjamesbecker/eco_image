import mysql_connector as con
import time
from datetime import datetime
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
query_base = """insert into deploy (Flag, CameraNumber, SiteID, USDS,
 DeployStart, DeployEnd, PictureEnd, BatteryBrand, BatteryType, StartBatteryAmp,
 EndBatteryAmp, Infostrip, MotionDetect, QualitySettings, TimeLapseSettings,
 MemoryUsed, USGSGage, Comments, FieldStaffDeploy, FieldStaffRetrieve)
 values """

query = ''
with open('../Data/DeployDataClean.csv', mode='r') as infile:
    reader = csv.reader(infile)
    infile.readline()

    values = list(reader)
    buffer_size = 10
    i = 0
    for row in values:
        # the indices of the columns whose types/formats need to be changed
        dates = [4, 5, 6]
        ints = [0, 1]
        floats = [9, 10, 15]

        # replace null values with None and change types
        for j in range(0, len(row)):
            if row[j].rstrip(' ') == '':
                row[j] = None
            elif j in dates:
                d = datetime.strptime(row[j], "%m/%d/%Y").date()
                ts = time.mktime(d.timetuple())
                row[j] = ts
            elif j in ints:
                row[j] = int(row[j])
            elif j in floats:
                row[j] = float(row[j])
        row.extend(["MC", "MC"])
        query += '(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s),'
        id.extend(row)
        i = i + 1
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

# stop timing
stop = time.clock()
