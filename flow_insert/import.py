import mysql_connector as con
from datetime import datetime
import time
import multiprocessing as mp
import os
import meta as m
import argparse

des = """ECO DB Import Tool"""

parser = argparse.ArgumentParser(description=des,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-i',
                    '--in_dir',
                    type=str,
                    help='image input directory')
parser.add_argument('-q',
                    '--q_dir',
                    type=str,
                    help='image quarantine directory')
parser.add_argument('-d',
                    '--d_dir',
                    type=str,
                    help='image delete directory')
parser.add_argument('-c',
                    '--cpus',
                    type=int,
                    help='number of processors')
parser.add_argument('-b',
                    '--buffer_size',
                    type=int,
                    help='buffer size')
parser.add_argument('-p',
                    '--port',
                    type=int,
                    help='DB port number')
parser.add_argument('-m',
                    '--move',
                    type=int,
                    help='move files and cleanup directories')
args = parser.parse_args()

if args.in_dir is not None and os.path.exists(args.in_dir):
    in_dir = args.in_dir
else:
    raise IOError
if args.q_dir is not None and os.path.exists(args.q_dir):
    q_dir = args.q_dir
else:
    raise IOError
if args.d_dir is not None and os.path.exists(args.d_dir):
    d_dir = args.d_dir
else:
    raise IOError
if args.cpus is not None:
    cpus = args.cpus
else:
    cpus = mp.cpu_count()
if args.buffer_size is not None:
    buffer_size = args.buffer_size
else:
    buffer_size = 1
if args.port is not None:
    port = args.port
else:
    port = 3306
if args.move is not None:
    move = args.move
else:
    move = 0


# Strips the returned meta of 'category' labels in the keys.
def cleaned_meta(meta):
    values = []
    for d in meta:
        for s in d:
            vals = {}
            for v in d[s]:
                tag = v.split(' ')
                if len(tag) > 1:
                    tag = tag[1]
                else:
                    tag = tag[0]
                vals[tag] = d[s][v]
            values.append(vals)
    return values


# Cleans out files and moves them to a given directory, deleting empty dir.
def clean_dir(out, files):
    for f in files:
        folder = os.path.basename(os.path.dirname(f))
        out_folder = os.path.join(out, folder)
        fi = os.path.join(out_folder, os.path.basename(f))
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        os.rename(f, fi)
        cur_folder = os.path.split(f)[0]
        if(cur_folder != in_dir):
            try:
                os.rmdir(cur_folder)
            except OSError:
                pass


if __name__ == '__main__':
    __spec__ = None
    # read cfg for host and database to connect to
    with open("db.cfg") as f:
        host = f.readline().strip('\n')
        database = f.readline().strip('\n')

    # read cfg for credentials (username and password to DB)
    # TODO: decrypt using a key found in another file(%s)
    with open("cred.cfg") as f:
        usr = f.readline().strip('\n')
        pwd = f.readline().strip('\n')

    # variable to be injected
    inj = []
    # failure injectable
    fail = []
    # the columsn matching the DB specification
    # and format (to be inserted into)
    columns = ("CameraNumber,Orientation,"
               "XResolution,YResolution,"
               "ResolutionUnit,Software,TimeTaken,"
               "ExposureTime,FNumber,ExposureProgram,"
               "ISOSpeedRatings,ExifVersion,"
               "ComponentsConfiguration,"
               "CompressedBitsPerPixel,"
               "ShutterSpeedValue,ApertureValue,"
               "ExposureBiasValue,"
               "MaxApertureValue,MeteringMode,Flash,"
               "Checksum,RawData")
    # exif tags that are returned (in this order)
    tags = ("ImageDescription,Orientation,"
            "XResolution,YResolution,"
            "ResolutionUnit,Software,DateTime,"
            "ExposureTime,FNumber,ExposureProgram,"
            "ISOSpeedRatings,ExifVersion,"
            "ComponentsConfiguration,"
            "CompressedBitsPerPixel,"
            "ShutterSpeedValue,ApertureValue,"
            "ExposureBiasValue,"
            "MaxApertureValue,MeteringMode,Flash,"
            "Checksum,RawData")
    # statement for image table insertion
    statement = "insert into jpgImages (" + columns + ") values "
    # build upon this to use for image insert ... (%s, %s,...),
    query = ''
    # statement and query for inserting into log table
    log = ("insert into importrun (Start, End, Attempted) values (%s, %s, %s);")
    # statement for inserting into failure table
    failure = ("insert into failure (Start, Checksum, Note) values ")
    # build upon this to use for failure insert ... (%s, %s,...)
    fail_query = ''

    # tags that should be taken from exif
    # (exactly as they are found from exifread)
    tag_set = ("Image ImageDescription, Image Orientation, "
               "Image XResolution, Image YResolution, "
               "Image ResolutionUnit, Image Software, "
               "Image DateTime, EXIF ExposureTime, EXIF FNumber, "
               "EXIF ExposureProgram, EXIF ISOSpeedRatings, "
               "EXIF ExifVersion, EXIF ComponentsConfiguration, "
               "EXIF CompressedBitsPerPixel, "
               "EXIF ShutterSpeedValue, "
               "EXIF ApertureValue, EXIF ExposureBiasValue, "
               "EXIF MaxApertureValue, "
               "EXIF MeteringMode, EXIF Flash")

    # from the single-value dict, get the total and metatags pair
    # then clean the data
    (total, meta), = m.get_meta(in_dir, tag_set, q_dir, cpus, move).items()
    values = cleaned_meta(meta)

    # list of the exif tags
    cols = tags.split(',')
    # current buffer size
    i = 0
    # count the number of times the buffer is dumped erroneously
    buffer_dumps = 0
    # the types to be converted and fitted to the right formats
    timestamps = ["DateTime"]
    ints = ["ImageDescription", "XResolution", "YResolution"]
    floats = ["ExposureTime",
              "FNumber",
              "CompressedBitsPerPixel",
              "ShutterSpeedValue",
              "MaxApertureValue"]
    # files that may be moved to out dir or quarantine dir
    # depends on if they are successful in insert
    files_pending = []
    # files to be moved to d_dir
    deletes = []
    # files to be moved to quarantine
    quarantines = []
    # performance timing
    start = time.clock()
    # epoch timing for log
    l_start = time.time()
    
    # connect and query
    with con.MYSQL(host, database, port, usr, pwd) as db:
        # for every row of exif
        for row in values:
            # row of exif data, this is what will be inserted
            # made of converted values
            fRow = []
            try:
                # for every exif tag that is in the returned meta dict
                for c in cols:
                    # do conversions
                    if row.get(c) is None or row[c].strip(' ') == '':
                        row[c] = None
                    elif c in timestamps:
                        d = datetime.strptime(row[c], "%Y:%m:%d %H:%M:%S")
                        ts = int(time.mktime(d.timetuple()))
                        row[c] = ts
                    elif c in ints:
                        row[c] = int(row[c])
                    elif c in floats:
                        if '/' in row[c]:
                            nums = row[c].split('/')
                            row[c] = float(nums[0])/float(nums[1])
                        else:
                            row[c] = float(row[c])
                    fRow.append(row[c])
            # on value conversion failure, put in quarantine and log failure
            except Exception as e:
                quarantines.append(row["Path"])
                fail.extend([l_start, row["Checksum"], str(e)[0:800]])
                fail_query += '(' + '%s,'*2 + '%s),'
                continue
            # buffer is added to if this point reached
            i = i + 1
            # add current file to pending
            files_pending.append(row["Path"])
            row = fRow
            # new image to insert, append to query
            query += '(' + '%s,'*(len(cols)-1) + '%s),'
            # new image to insert, append to value injectable
            inj.extend(row)
            # when buffer is full, attempt insert
            if i >= buffer_size:
                # attempt query
                query = statement + query[:-1] + ';'
                r = db.query(query, inj)
                # on success, add file to be moved to deleted folder
                # reset pending files
                if not db.data_errors:
                    deletes.extend(files_pending)
                # on failure
                else:                    
                    # reset general query errors
                    db.data_errors = ''
                    # buffer dumped, increment
                    buffer_dumps += 1
                    # break up the injectables list into lists corresponding to files
                    # each sublist length matches the number of columns
                    data = [inj[x:x+len(cols)] for x in range(0, len(inj), len(cols))]
                    # single-insert query
                    query = statement + '(' + '%s,'*(len(cols)-1) + '%s);'
                    # perform single-inserts using sublists
                    # insert failures and unmasked error codes
                    for d, j in zip(data, files_pending):
                        inj = d
                        r = db.query(query, inj)
                        if not db.data_errors:
                            deletes.extend(j)
                        else:
                            quarantines.append(j)
                            fail.extend([l_start, d[20], str(db.data_errors)[0:800]])
                            fail_query += '(' + '%s,'*2 + '%s),'
                            db.data_errors = ''
                # no more files pending
                files_pending = []
                # buffer is reset
                i = 0
                # injectables is reset
                inj = []
                # query is reset
                query = ''
        # reached end of files, buffer not empty, attempt to insert
        # same process as above
        if i != 0:
            query = statement + query[:-1] + ";"
            r = db.query(query, inj)
            if not db.data_errors:
                deletes.extend(files_pending)
                files_pending = []
            else:
                db.data_errors = ''
                buffer_dumps += 1
                data = [inj[x:x+len(cols)] for x in range(0, len(inj), len(cols))]
                query = statement + '(' + '%s,'*(len(cols)-1) + '%s);'
                for d, j in zip(data, files_pending):
                    inj = d
                    r = db.query(query, inj)
                    if not db.data_errors:
                        deletes.extend(j)
                    else:
                        quarantines.append(j)
                        fail.extend([l_start, d[20], str(db.data_errors)[0:800]])
                        fail_query += '(' + '%s,'*2 + '%s),'
                        db.data_errors = ''
        # done inserting images, record performance end
        stop = time.clock()
        # if images have actually been processed, insert into failure, performance
        if total > 0:
            l_stop = time.time()
            db.query(log, [l_start, l_stop, total])
            if fail:
                fail_query = failure + fail_query[:-1] + ";"
                r = db.query(fail_query, fail)

            r = db.query("insert into performance (Start, Time, Buffer, Attempted, BufferDumps) values (%s, %s, %s, %s, %s);", [l_start, round(stop-start, 6), buffer_size, total, buffer_dumps])
            if move:
                clean_dir(d_dir, deletes)
                clean_dir(q_dir, quarantines)
