import glob
import os

flow_label = '/media/data/flowlabel_clean_for_analysis_100719.csv'
in_dir = '/media/data/flow_2018_sites'
with open(flow_label,'r') as f:
    data = [line.replace('\r','').replace('\n','').split(',') for line in f.readlines()]
    header,data,C,S,SC = data[0],data[1:],{},{},{}
    for row in data:
        file,sid,cat = '/'+row[0]+'/'+row[1],int(row[2]),int(row[3])
        if sid in S: S[sid] += [file]
        else:        S[sid]  = [file]
        if cat in C: C[cat] += [file]
        else:        C[cat]  = [file]
        if sid in SC:
            if cat in SC[sid]: SC[sid][cat] += [file]
            else:              SC[sid][cat]  = [file]
        else:
            SC[sid] = {cat:[file]}

for c in C:
    for i in range(len(C[c])):
        in_path  = in_dir+C[c][i]
        out_path = in_dir+'/label_%s/'%c+C[c][i].rsplit('/')[-1]
        if not os.path.exists(in_dir+'/label_%s/'%c): os.mkdir(in_dir+'/label_%s/'%c)
        if os.path.exists(in_path): os.rename(in_path,out_path)