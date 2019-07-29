import os
import piexif

def update_exif_descrip (path,cid,tag='0th',byte=306):
    E = piexif.load(path)
    E[tag][byte] = cid
    piexif.insert(piexif.dump(E),path)
    return True
    

imagedir = 'S:\\Streamflow\\Dry Stream Documentations\\2018 Streamflow Monitoring\\Pictures'

##read raw text
with open('P:\\Projects\\GitHub_Prj\\eco_image\\data\\cameraID.csv','r') as f: raw = f.readlines()

##create dictionary to store data
fpath_cameraID = {'fpath':[],'cameraid':[]}

for i in range(len(raw)):
    line = raw[i]
    p1 = line.find(',')
    p2 = line.find('\n')
    fpath = os.path.join(imagedir, line[0:p1])
    cameraid = line[(p1+1):p2]
    fpath_cameraID['fpath'].append(fpath)
    fpath_cameraID['cameraid'].append(cameraid)   
    
#Create an empty list and store all JPG files in the directory
files=[]
for f in range(len(fpath_cameraID['fpath'])):
    fpathfiles = os.listdir(fpath_cameraID['fpath'][f])
    for i in range(len(fpathfiles)):
           if fpathfiles[i].endswith(".JPG"):
             cid = fpath_cameraID['cameraid'][f]
             path = os.path.join(fpath_cameraID['fpath'][f],fpathfiles[i])   #update exif description
                )
    
E = piexif.load(path)