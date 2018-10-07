import sys
import os
import glob
import cv2
import pytesseract
import utils

im_path = utils.local_path()+'/data/MOULTRIE_M888_CROP_FULL.JPG'
config = ('-l eng --oem 1 --psm 3')
im = cv2.imread(im_path,cv2.IMREAD_COLOR)
text = pytesseract.image_to_string(im, config=config)
print(text)