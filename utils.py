import os
import sys
import exifread
import piexif

def local_path():
    return os.path.abspath(__file__).replace('utils.py','')

#given an image file with exif metadat return set of the tags that are required
def get_exif_tags(path,tag_set='all'):
    tags,T = {},{}
    with open(path,'rb') as f: tags = exifread.process_file(f)
    if tag_set=='all': tag_set = set(tags.keys())
    for t in tags:
        if t in tag_set: T[t] = str(tags[t].values.rstrip(' '))
    return T
