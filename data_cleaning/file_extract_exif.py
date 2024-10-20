import os
import glob
import piexif
from datetime import datetime
unix = False
os_dir_char = ('/' if unix else '\\')
prefered_ext = 'JPG'
from data_cleaning import mysql_connector as msc

cf_dir = ''
db_scm = 'strc'

# Extract relevant exif data by specifying exif tag & byte code.  cd 306 = date_time cd 270 = title (stream connect cat)
def get_exif(path,tag='0th',byte=270):
    E = piexif.load(path)
    t = ''
    if tag in E:
        if byte in E[tag]:
            t = E[tag][byte].decode('ascii')
    return t

def parse_exif_date_time(t):
    return datetime.strptime(t,'%Y:%m:%d %H:%M:%S')

with open(cf_dir, 'r') as f:
    s = f.read()
    f.close()
config = [line.rsplit(',') for line in s.rsplit('\n')]  # chop up the text by the newline='\n and the delim
config_uid = config[0][1]
config_pw = config[1][1]

imagedir = 'D:\\flow_data\\Trail_Camera_Pictures'

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
sample_err = []

for i in range(len(jpgs)):
    try:
        f_nm = os.path.splitext(os.path.basename(jpgs[i]))[0]
        s_data = f_nm.rsplit("_")
        s_dttm = parse_exif_date_time(get_exif(jpgs[i], '0th', 306))
        s_date = s_dttm.strftime('%Y-%m-%d %H:%M:%S')
        # s_time = s_dttm.strftime('%H:%M:%S')
        s_catg = get_exif(jpgs[i], '0th', 270)
        if s_catg=='':
            sample_err += [jpgs[i]]
        else:
            s = [f_nm] + s_data + [s_date] + [s_catg]
            sample_img += [s]
    except Exception as e:
        print(e)
        sample_err += [e]

fpath_err = 'D:\\flow_data\\Trail_Camera_Pictures\\err.txt'
s = '\n'.join([str(sample_err[i])][0]for i in range(len(sample_err)))
with open(fpath_err, 'w') as f:
                f.write(s)

###In Progress - May want to just get above in DB to make easier to query.  Add img hash?###
sql_insert = 'INSERT INTO ' + db_scm + \
             '.flowlabel (`fileNm`,`staSeq`,`CameraID`,`UserID`,`fileN`,`TimeTaken`,`Category`,`insDate`) ' \
             'VALUES (?,?,?,?,?,?,?,?);'
sql_err_log = 'INSERT INTO ' + db_scm + '.errlog VALUES (?,?,?);'

l_err = []

try:
    ins_date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    with msc.MYSQL('localhost', db_scm, 3306, config_uid, config_pw) as dbo:
        for i in range(len(sample_img)):
            if sample_img[i] is not None and len(sample_img[i]) == 10:
                    v_insert = sample_img[i][0:2] + sample_img[i][5:10] + [ins_date]
                    ins = dbo.query(sql_insert,v_insert)
                    if ins != {}:
                        print('error with img %s err=%s' % (sample_img[i][0], ins[sorted(ins)[0]]))
                        err = [sample_img[i][0],ins[sorted(ins)[0]],ins_date]
                        db_err = dbo.query(sql_err_log,err)
                    else:
                        print('success with img %s n %s' % (sample_img[i][0], i))
            else:
                l_err += [sample_img[i]]
except TypeError as e:
    print(e)

lpath_err = 'D:\\flow_data\\Trail_Camera_Pictures\\l_err.txt'
s = '\n'.join([','.join([str(e) for e in row]) for row in l_err])
with open(lpath_err, 'w') as f:
    f.write(s)

sql_select = 'SELECT * FROM flowlabel;'
with msc.MYSQL('localhost', db_scm, 3306, config_uid, config_pw) as dbo:
    sel = dbo.query(sql_select)

