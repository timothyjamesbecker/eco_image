import glob
import os

#flow_label should have headers and be a CSV
flow_label = '/media/tbecker/g_data/flowlabel_clean_for_analysis_100719.csv'
in_dir     = '/media/tbecker/g_data/2018/'
with open(flow_label,'r') as f:
    data = [line.replace('\r','').replace('\n','').split(',') for line in f.readlines()]
    header,data,C,S,SC = data[0],data[1:],{},{},{}
    hidx = {header[i]:i for i in range(len(header))}
    for row in data:
        file    = '/'+row[hidx['folder']]+'/'+row[hidx['fpath']]
        sid,cat = int(row[hidx['sid']]),int(row[hidx['category']])
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