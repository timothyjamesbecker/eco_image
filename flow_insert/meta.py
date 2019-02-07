import os
import time
import multiprocessing as mp
import utils
from collections import defaultdict
import hashlib

"""ImageDescription, Orientation,
XResolution, YResolution,
ResolutionUnit, Software, DateTime,
ExposureTime, FNumber, ExposureProgram,
ISOSpeedRatings, ExifVersion,
ComponentsConfiguration, CompressedBitsPerPixel,
ShutterSpeedValue, ApertureValue, ExposureBiasValue,
MaxApertureValue, MeteringMode, Flash,
Checksum, Data"""

des = """image meta data crawler"""

# holds gathered results from async job
results = []


# extends result list with a new result
def collect_results(result):
    for r in result:
        for k in r:
            data = b''
            checksum = ''
            hasher = hashlib.sha256()
            with open(r[k]["Path"], 'rb') as f:
                for chunk in iter(lambda: f.read(524288), b""):
                    data += chunk
                    hasher.update(chunk)
            checksum = hasher.hexdigest()
            r[k]["RawData"] = str(data)
            r[k]["Checksum"] = checksum
    results.extend(result)


# for each image in this SID collection
# get the corresponding tag/value dict
# and pair the dict to the filename.
# append resulting dict to S
def exif(sid_images, tag_set):
    S = []
    for path in sid_images:
        if os.name == 'nt':
            p = path.rsplit('\\')[-1]
            vals = utils.get_exif_tags(path, tag_set)
            S.append({p: vals})
        else:
            p = path.rsplit('/')[-1]
            vals = utils.get_exif_tags(path, tag_set)
            S.append({p: vals})
    # format: [{file1: dict of meta k:v for file1},
    # {file2: dict of meta k:v for file2}, ...]
    return S


# cleans out files and moves them to a given directory, deleting empty dir.
# transplants dir structure to new folder, mimicking original in_dir structure.
def clean_dir(out, in_dir, files):
    for f in files:
        folder = os.path.basename(os.path.dirname(f))
        out_folder = os.path.join(out, folder)
        fi = os.path.join(out_folder, os.path.basename(f))
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        os.rename(f, fi)
        cur_folder = os.path.split(f)[0]
        if(cur_folder != in_dir):
            # Can only del if empty!
            try:
                os.rmdir(cur_folder)
            except OSError:
                pass


# gathers the needed tag:values for each file in the in_dir
# ensures that files fit folder naming structure before processing
# and that files are in-fact .jpg images.
def get_meta(in_d, tag_s, q_d, c=mp.cpu_count(), m=0):
    if in_d is not None and os.path.exists(in_d):
        in_dir = in_d
    else:
        raise IOError
    if tag_s is not None:
        tag_set = tag_s.replace("\n", "").split(', ')
    else:
        tag_set = []
    if q_d is not None:
        if os.path.exists(q_d):
            q_dir = q_d
        else:
            os.mkdir(q_d)
    if c is not None:
        cpus = c
    else:
        cpus = mp.cpu_count()
    if m is not None:
        move = m
    else:
        move = 0

    # holds images that are to be moved
    # to a quarantine folder and NOT returned.
    quarantines = []

    # paths of all images.
    # form of {SID:[dirs that belong to that SID], ...}
    paths = defaultdict(lambda: [])
    for dirpath, dirnames, filenames in os.walk(in_dir):
        if dirnames:
            for d in dirnames:
                p = str(d.rsplit('_')[0])
                paths[p].append(os.path.join(dirpath, d))
    # map/scatter---------------------------------------------------------------------------------------------------
    p1 = mp.Pool(processes=cpus)
    # keep running total of all processed images.
    # these are images that will (be attempted to) be inserted into DB.
    total = 0
    # for every SID:
    for sid in paths:
        # for every path belonging to SID:
        for p in paths[sid]:
            # split folder name to later match against file name
            folder_info = os.path.basename(p).split('_')
            folder_info[-1] = folder_info[-1].split(' ')[0]
            # for every path, name, file in the path
            for dirpath, dirnames, filenames in os.walk(p):
                # images belonging to SID
                sid_images = []
                # for every file
                for f in filenames:
                    # ensure that it is jpg, and match name to folder name
                    if f.lower().endswith(".jpg"):
                        file_info = f.split('_')
                        if ' ' in file_info[-1]:
                            file_info[-1] = file_info[-1].split(' ')[0]
                        if bool(set(file_info) == set(folder_info)):
                            sid_images.append(os.path.join(dirpath, f))
                        else:
                            quarantines.append(os.path.join(dirpath, f))
                # add based on added files
                total += len(sid_images)
                # only attempt to dispatch processes if
                # there is something to process
                if len(sid_images) > 0:
                    p1.apply_async(exif,
                                   args=(sid_images, tag_set),
                                   callback=collect_results)
                    time.sleep(0.1)
    p1.close()
    p1.join()
    if move:
        clean_dir(q_dir, in_dir, quarantines)

    # format: {num processed: [{file1: dict of meta k:v for file1}, ...]}
    return {total: results}
