import utils
import piexif
import exifread

def deg_min_sec_to_dec(d):
    return d[0].num+(d[1].num/60.0)+((1.0*d[2].num)/(1.0*d[2].den))/3600.00

path = '/home/tbecker/Desktop/GoProHero7/GOPR0074.JPG'
with open(path,'rb') as f:
    tags = exifread.process_file(f)
lat,lon = 0.0,0.0
if tags.has_key('GPS GPSLatitude'):
    lat = deg_min_sec_to_dec(tags['GPS GPSLatitude'].values)
if tags.has_key('GPS GPSLongitude'):
    lon = deg_min_sec_to_dec(tags['GPS GPSLongitude'].values)