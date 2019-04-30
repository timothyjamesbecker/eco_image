import os
import csv

#identify a directory and list only folders in the directory
imagedir = 'S:/Streamflow/Dry Stream Documentations/2018 Streamflow Monitoring/Pictures'
fpathdir = [d for d in os.listdir(imagedir) if os.path.isdir(os.path.join(imagedir, d))]

#Create an empty list and store only folders need for analysis
fpathdirA = []
for i in range(len(fpathdir)):
    if len(fpathdir[i]) > 16 and len(fpathdir[i]) > 28:
        fpathdirA.append(fpathdir[i])

#Create an empty list and store all JPG files in the directory
files=[]
for f in range(len(fpathdirA)):
    fpathfiles = os.listdir(os.path.join(imagedir,fpathdirA[f]))
    for i in range(len(fpathfiles)):
           if fpathfiles[i].endswith(".JPG"):
                files.append(fpathfiles[i])

#Find the nth position of a character in a string
def find_nth(char,id, n):
    start = char.find(id)
    while start >= 0 and n > 1:
        start = char.find(id, start+len(id))
        n -= 1
    return start


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

#datadict = {'fpath':fpath,'sid':sid,'name':name}

#Write list to csv
#with open('filetest.csv', 'wb') as csvfile:
    #writer = csv.writer(csvfile, delimiter=' ',quoting=csv.QUOTE_NONE)
    #writer.writerow(fpath + ',' + sid + ',' + name + ',' + sdate + ',' + edate)
    
#Write dictionary to csv
writefile = 'testfiledict.csv'
fieldnames = ['fpath','sid', 'name']
with open( writefile, 'w' ) as f:
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    writer.writerows(zip(imgfiles['fpath'],imgfiles['sid'],imgfiles['name'],imgfiles['sdate'],imgfiles['edate']))
    


    



