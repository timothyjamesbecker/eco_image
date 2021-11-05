import os
import glob
import piexif
import datetime
import numpy as np
import time
unix = False
os_dir_char = ('/' if unix else '\\')
prefered_ext = 'JPG'

# Extract relevant exif data by specifying exif tag & byte code.  cd 306 = date_time cd 270 = title (stream connect cat)
def get_exif(path,tag,byte):
    E = piexif.load(path)
    t = E[tag][byte].decode('ascii')
    return t

def parse_exif_date_time(t):
    return datetime.datetime.strptime(t,'%Y:%m:%d %H:%M:%S')

imagedir = 'C:\\Users\\deepuser\\Documents\\Projects\\Trail_Cam_Imgs'

#[1] get all valid JPGS that live in a sub dir-------------------------------------
if unix:
    jpgs =  glob.glob(imagedir+'%s*%s*.JPG'%(os_dir_char,os_dir_char))+\
            glob.glob(imagedir+'%s*%s*.JPEG'%(os_dir_char,os_dir_char))+\
            glob.glob(imagedir+'%s*%s*.jpg'%(os_dir_char,os_dir_char))+\
            glob.glob(imagedir+'%s*%s*.jpeg'%(os_dir_char,os_dir_char))
else:
   jpgs =   glob.glob(imagedir+'%s*%s*.JPG'%(os_dir_char,os_dir_char))+\
            glob.glob(imagedir+'%s*%s*.JPEG'%(os_dir_char,os_dir_char))

sample_img = []

for i in range(len(jpgs)):
    f_nm = os.path.splitext(os.path.basename(jpgs[i]))[0]
    s_data = f_nm.rsplit("_")
    s_dttm = parse_exif_date_time(get_exif(jpgs[i], '0th', 306))
    s_date = s_dttm.strftime('%Y-%m-%d')
    s_time = s_dttm.strftime('%H:%M:%S')
    s_catg = get_exif(jpgs[i], '0th', 270)

    s = [f_nm] + s_data + [s_dttm] + [s_date] + [s_time] +[s_catg]

    sample_img += [s]


###In Progress - May want to just get above in DB to make easier to query.  Add img hash?###
sid = sample_img[0][1]
cid = sample_img[0][1]


categories = [int(sample_img[i][11]) for i in range(len(sample_img))]
np.mean(categories)



