import os
import copy
import time
import pytz
import glob
import piexif
import utils
import datetime

def get_exif_date_time(path,tag='0th',byte=306):
    E = piexif.load(path)
    t = E[tag][byte]
    return parse_exif_date_time(t)

def parse_exif_date_time(t):
    return datetime.datetime.strptime(t,'%Y:%m:%d %H:%M:%S')

def update_exif_date_time_hours(path,tag='0th',byte=306,offset=-1):
    E = piexif.load(path)
    t = parse_exif_date_time(E[tag][byte])
    o = datetime.timedelta(hours=offset)
    t += o
    E[tag][byte] = t.strftime('%Y:%m:%d %H:%M:%S')
    piexif.insert(piexif.dump(E),path)
    return True

def get_dst_change_points(y,zone='America/New_York'):
    tz = pytz.timezone(zone)
    trans = tz._utc_transition_times
    T = []
    for t in trans:
        if t.date().year == y:
            T += [t]
    return sorted(T)

def is_in_dst(t,T):
    if t >= T[0] and t <= T[1]:
        return True
    else:
        return False          

path = '/Users/tbecker/Documents/Projects/GitHubProjects/eco_image/data/'+\
       'camera_examples/14523_BurtonBrook_051018_062018/14523_BurtonBrook_051018_062018 (13).JPG'

E = piexif.load(path)
t1 = E['0th'][306]

e1 = E['Exif'][36867]
e2 = E['Exif'][36868]

fpath = os.path.dirname(path)
sdep = datetime.datetime.strptime(fpath[-13:-7],'%m%d%y')

if is_in_dst(sdep,T):
    update_exif_date_time_hours(path,tag='0th',byte=306,offset=-1)

#read raw text
with open('/Users/tbecker/Desktop/data_test.tsv','r') as f: raw = f.readlines()

#make some data structures and chop up your data rows
cols,data = {},[]
for i in range(len(raw)):
    if i==0:
        l = raw[i].replace('\n','').split('\t')
        cols = {l[j]:j for j in range(len(l))}
    else:
        data += [raw[i].replace('\n','').split('\t')]
        
    
