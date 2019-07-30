import os
import glob

def rename(dir, ftype, titlePattern):
    for pathAndFilename in glob.iglob(os.path.join(dir, ftype)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        os.rename(pathAndFilename, 
                  os.path.join(dir, titlePattern % title + ext))
        
rename(r'c:\temp\xx', r'*.doc', r'new(%s)')

imagedir = 'P:\\Projects\\2018\\FlowImpair\\TrailCamFlowImageDataPrj\\DataCleaning\\testingpics'

filelist = []

for i in range(len(os.listdir(imagedir))):
    dirfolder = os.listdir(imagedir)[i]
    if os.path.isdir(os.path.join(imagedir,dirfolder)):
        fpath = os.path.join(imagedir,dirfolder)
        files = os.listdir(fpath)
    for n in range(len(files)):
        if files[n].endswith(".JPG"):
            filepath = os.path.join(fpath,files[n])
            filelist.append(filepath)
        
for n in range(len(filelist)):
    filedirectory = os.path.dirname(filelist[n])
    newfilepath = os.path.dirname(filelist[n])+'\\'+'_'+str(i)+".JPG"
    os.rename(filelist[n],newfilepath)
            
            
            try:
                os.rename(filepath,newfilepath)
            except WindowsError:
                os.remove(newfilepath)
                os.rename(filepath,newfilepath)

path =  'P:\\Projects\\2018\\FlowImpair\\TrailCamFlowImageDataPrj\\DataCleaning\\testingpics\\15240_HubbardRiver_040218_040218_44'
filenames = []
for i in range(len(os.listdir(path))):
    if os.listdir(path)[i].endswith(".JPG"):
        filenames.append(os.path.join(path,os.listdir(path)[i]))

for i in range(len(filenames)):
    foldername=os.path.basename(os.path.dirname(filenames[0]))
    newname=os.path.dirname(os.path.dirname(filenames[filename]))
    os.rename(filename, filename.replace(" ", "-").lower())
                


