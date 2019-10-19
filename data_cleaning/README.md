# Image File / Directory Data Cleaning Tools

Python 2.7.15

'rename_files_in_imagefolder.py' - identifies folders with jpg images and renames
jpg files with correct naming convention for flow_observation DB as 
long as folders were named correctly.  Basic ver working.  Will update to work 
via command line.

'exif_time_write_test.py' - identifies deployments where time was set during
daylight savings and converts to eastern standard time and then to POSIX for 
DB.  Basic ver working.  Updates being made to work via command line.

'update_exif_cameraID.py' - extracts camera ID from file (for 2018 data) or
from folder name (for 2019 data and beyond) and update exif description tag

'file_parseToupdate_cameraID.py' - extracts data from file names and exif data
to compile table to update cameraID for 2018 data
