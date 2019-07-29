import os
import csv
import piexif
import datetime
import time

#identify a directory and list only folders in the directory
imagedir = 'S:\\Streamflow\\Dry Stream Documentations\\2018 Streamflow Monitoring\\Pictures'
fpathdir = [d for d in os.listdir(imagedir) if os.path.isdir(os.path.join(imagedir, d))]

def get_exif_date_time(path,tag='0th',byte=306):
    E = piexif.load(path)
    t = E[tag][byte]
    return t

def parse_exif_date_time(t):
    return datetime.datetime.strptime(t,'%Y:%m:%d %H:%M:%S')

#Find the nth position of a character in a string
def find_nth(char,id, n):
    start = char.find(id)
    while start >= 0 and n > 1:
        start = char.find(id, start+len(id))
        n -= 1
    return start

#Create an empty list and store only folders need for analysis
fpathdirA = []
for i in range(len(fpathdir)):
    if len(fpathdir[i]) > 16 and len(fpathdir[i]) > 28:
        fpathdirA.append(fpathdir[i])
        
imgfiledir = {'fpathdir':[],'sid':[],'name':[],'sdate':[],'edate':[]}
for n in range(len(fpathdirA)):
    fpath = fpathdirA[n]
    p1 = find_nth(fpath,'_',2)
    p2 = find_nth(fpath,'_',3)
    sid = fpath[0:5]
    name =  fpath[fpath.find('_')+1:p1]
    sdate = fpath[p1+1:p2]
    edate = fpath[p2+1:]
    imgfiledir['fpathdir'].append(fpath)
    imgfiledir['sid'].append(sid)
    imgfiledir['name'].append(name)
    imgfiledir['sdate'].append(sdate)
    imgfiledir['edate'].append(edate)

#Write imgfiledir dictionary to csv
writefile = 'imgfiledir2018_attributeinfo.csv'
fieldnames = ['fpath','sid', 'name','sdate','edate']
with open( writefile, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    writer.writerows(zip(imgfiledir['fpathdir'],imgfiledir['sid'],imgfiledir['name'],imgfiledir['sdate'],imgfiledir['edate']))

#Create an empty list and store all JPG files in the directory
files=[]
for f in range(len(fpathdirA)):
    fpathfiles = os.listdir(os.path.join(imagedir,fpathdirA[f]))
    for i in range(len(fpathfiles)):
           if fpathfiles[i].endswith(".JPG"):
                files.append(fpathfiles[i])

#Create a dictionary and store attributes needed to link to camera id
imgfiles = {'fpath':[],'sid':[],'name':[],'sdate':[],'edate':[]}

for n in range(len(files)):
    fpath = files[n]
    p1 = find_nth(fpath,'_',2)
    p2 = find_nth(fpath,'_',3)
    sid = fpath[0:5]
    name =  fpath[fpath.find('_')+1:p1]
    sdate = fpath[p1+1:p2]
    edate = fpath[p2+1:fpath.find(' ')]
    data = fpath + ',' + sid + ',' + name + ',' + sdate + ',' + edate
    imgfiles['fpath'].append(fpath)
    imgfiles['sid'].append(sid)
    imgfiles['name'].append(name)
    imgfiles['sdate'].append(sdate)
    imgfiles['edate'].append(edate)
    
#Write dictionary to csv
writefile = 'imgfiles2018_attributeinfo.csv'
fieldnames = ['fpath','sid', 'name','sdate','edate']
with open( writefile, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    writer.writerows(zip(imgfiles['fpath'],imgfiles['sid'],imgfiles['name'],imgfiles['sdate'],imgfiles['edate']))


##Get the full file path of all files for analysis
##Extract the exif datetime and needed into to idenfify camera id and store in dictionary
    
filepath=[]
for f in range(len(fpathdirA)):
    fpathfiles = os.listdir(os.path.join(imagedir,fpathdirA[f]))
    for i in range(len(fpathfiles)):
           if fpathfiles[i].endswith(".JPG"):
                filepath.append(os.path.join(imagedir,fpathdirA[f],fpathfiles[i]))


imgfiles = {'pathtofile':[],'folder':[],'fpath':[],'sid':[],'name':[],'sdate':[],'edate':[],'fdatetime':[],'nighttime':[]}

#Par
#for n in range (len(filepath)):
for n in range (len(filepath)):
    p=find_nth(filepath[n],'\\',6)+1
    fpath = filepath[n][p:len(filepath)]
    p1 = find_nth(fpath,'_',2)
    p2 = find_nth(fpath,'_',3)
    p3 = find_nth(filepath[n],'\\',5)+1
    folder=filepath[n][p3:p-1]
    sid = fpath[0:5]
    name =  fpath[fpath.find('_')+1:p1]
    sdate = fpath[p1+1:p2]
    edate = fpath[p2+1:fpath.find(' ')]
    fdatetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(filepath[n])))
    nighttime = int(fdatetime[11:13]) > 4 and int(fdatetime[11:13]) <21
    imgfiles['fpath'].append(fpath)
    imgfiles['sid'].append(sid)
    imgfiles['name'].append(name)
    imgfiles['sdate'].append(sdate)
    imgfiles['edate'].append(edate)
    imgfiles['pathtofile'].append(filepath[n])
    imgfiles['fdatetime'].append(fdatetime)
    imgfiles['nighttime'].append(nighttime)
    imgfiles['folder'].append(folder)
    
writefile = 'imgfiles2018_attributeinfo.csv'
fieldnames = ['pathtofile','folder','fpath','sid', 'name','sdate','edate','fdatetime','nighttime']
with open( writefile, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    writer.writerows(zip(imgfiles['pathtofile'],imgfiles['folder'],imgfiles['fpath'],imgfiles['sid'],imgfiles['name'],imgfiles['sdate'],imgfiles['edate'],imgfiles['fdatetime'],imgfiles['nighttime']))
    

    
#datadict = {'fpath':fpath,'sid':sid,'name':name}

#Write list to csv
#with open('filetest.csv', 'wb') as csvfile:
    #writer = csv.writer(csvfile, delimiter=' ',quoting=csv.QUOTE_NONE)
    #writer.writerow(fpath + ',' + sid + ',' + name + ',' + sdate + ',' + edate)
    


    



