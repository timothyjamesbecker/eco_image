import utils
#read image folder and name data and chop it to produce a file to camera map
image_file_path          = utils.local_path()+'/data/image_files.txt'
camera_deploy_path       = utils.local_path()+'/data/deploydata.csv'
camera_image_map_path    = utils.local_path()+'/data/camera_image_map.csv'
s,D,F = 'file_path,CID\n',{},{}

with open(image_file_path,'r') as f:    #change to +[l] for full path
    L = [l.rsplit('/')[-2].rsplit('_')+[l] for l in f.read().rsplit('\n')]
    for l in L: #(SID,d_start,d_stop)=>image_name
        k = tuple(l[:4])
        if F.has_key(k): F[k] += [l[4]]
        else:            F[k]  = [l[4]]
    
with open(camera_deploy_path,'r') as f:
    L = [l.replace('\n','').rsplit(',') for l in f.readlines()]
    h = {L[0][i]:i for i in range(len(L[0]))}
    L = L[1:]
    
for i in range(len(L)):
    k = (L[i][h['SID']],L[i][h['Sname']],L[i][h['Start_Date']],L[i][h['End_Date']])
    if F.has_key(k):
        for l in F[k]: s += ','.join([l,L[i][h['Camera_Number']]])+'\n'

with open(camera_image_map_path,'w') as f:
    f.write(s)