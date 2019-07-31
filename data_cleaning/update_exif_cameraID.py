import os
import piexif

#Find the nth position of a character in a string
def find_nth(char,id, n):
    start = char.find(id)
    while start >= 0 and n > 1:
        start = char.find(id, start+len(id))
        n -= 1
    return start

#Update exif description tag
def update_exif_descrip (path,cid,tag='0th',byte=270):
    E = piexif.load(path)
    E[tag][byte] = cid
    piexif.insert(piexif.dump(E),path)
    return True
    
# imagedir = 'P:\\Projects\\2018\\FlowImpair\\TrailCamFlowImageDataPrj\\DataCleaning\\testingpics'
imagedir = ''

##create dictionary to store fpath and cameraid
fpath_cameraID = {'fpath':[],'cameraid':[]}

#read raw text - 2018 camera ID link to folder
camera_id_test = 'P:\\Projects\\GitHub_Prj\\eco_image\\data\\cameraID_test.csv'
# camera_id_test = ''
with open(camera_id_test,'r') as f: raw = f.readlines()

## store 2018 data without new file naming convention
for i in range(len(raw)):
    line = raw[i]
    p1 = line.find(',')
    p2 = line.find('\n')
    fpath = os.path.join(imagedir, line[0:p1])
    cameraid = line[(p1+1):p2]
    fpath_cameraID['fpath'].append(fpath)
    fpath_cameraID['cameraid'].append(cameraid)

## store camerid for 2019 and beyond data using new file naming convention
for i in range(len(os.listdir(imagedir))):
    folder = os.listdir(imagedir)[i]
    fpath = os.path.join(imagedir,folder)
    p1 = find_nth(folder,'_',4)
    cameraid = folder[p1+1:]
    fpath_cameraID['fpath'].append(fpath)
    fpath_cameraID['cameraid'].append(cameraid)

    
#update exif 'title' with camera ID
for f in range(len(fpath_cameraID['fpath'])):
    fpathfiles = os.listdir(fpath_cameraID['fpath'][f])
    for i in range(len(fpathfiles)):
           if fpathfiles[i].endswith(".JPG"):
             cid = fpath_cameraID['cameraid'][f]
             path = os.path.join(fpath_cameraID['fpath'][f],fpathfiles[i])   
             update_exif_descrip (path,cid,tag='0th',byte=270) #update exif description
    