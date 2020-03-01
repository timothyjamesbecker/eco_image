import glob
import os
import argparse

des="""
---------------------------------------------------
Flow Label Mapper
Timothy James Becker 10-19-19 to 02-28-20
---------------------------------------------------
Given flow label csv file and input director,
Builds new folders and moves all files that
map to those labels which will prepare the
data for categorical based ML"""
parser = argparse.ArgumentParser(description=des.lstrip(" "),formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--in_dir',type=str,help='input directory of images\t[None]')
parser.add_argument('--flow_label',type=str,help='flow label CSV file\t[None]')
args = parser.parse_args()

#flow_label should have headers and be a CSV
flow_label = args.flow_label
in_dir     = args.in_dir
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