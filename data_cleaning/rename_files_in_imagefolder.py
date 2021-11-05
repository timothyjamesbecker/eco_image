import os
import glob
unix = False
os_dir_char = ('/' if unix else '\\')
prefered_ext = 'JPG'

imagedir = 'C:\\Users\\deepuser\\Documents\\Projects\\Trail_Cam_Imgs'
# dirs = ['home','mkozlak','Documents','Projects','GitHub','eco_image','data_cleaning','testdata']
# os_dir_char.join(dirs)

#[1] get all valid JPGS that live in a sub dir-------------------------------------
if unix:
    jpgs =  glob.glob(imagedir+'%s*%s*.JPG'%(os_dir_char,os_dir_char))+\
            glob.glob(imagedir+'%s*%s*.JPEG'%(os_dir_char,os_dir_char))+\
            glob.glob(imagedir+'%s*%s*.jpg'%(os_dir_char,os_dir_char))+\
            glob.glob(imagedir+'%s*%s*.jpeg'%(os_dir_char,os_dir_char))
else:
   jpgs =   glob.glob(imagedir+'%s*%s*.JPG'%(os_dir_char,os_dir_char))+\
            glob.glob(imagedir+'%s*%s*.JPEG'%(os_dir_char,os_dir_char))


#[2] associate directories to files
D = {}
for jpg in jpgs:
    dir = os_dir_char.join(jpg.rsplit(os_dir_char)[:-1])+os_dir_char
    if dir in D: D[dir] += [jpg]
    else:        D[dir]  = [jpg]
#[3] rename files
for dir in D:
    base = dir.rsplit(os_dir_char)[-2]
    for i in range(len(D[dir])): #for every jpg in the dir
        os.rename(D[dir][i],imagedir+os_dir_char+base+os_dir_char+base+'_%s.%s'%(i+1,prefered_ext))
        print(D[dir][i])
        print(imagedir+os_dir_char+base+os_dir_char+base+'_%s.%s'%(i+1,prefered_ext))

##Might need for windows
# try:
#     os.rename(filepath, newfilepath)
# except WindowsError:
#     os.remove(newfilepath)
#     os.rename(filepath, newfilepath)
                


