#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import glob
import cv2
import pytesseract
import utils

paths  = [utils.local_path()+'/data/processed_images/MOULTRIE_M888_CROP_FULL.JPG',
          utils.local_path()+'/data/processed_images/SPYPOINT_FORCE_CROP_FULL.JPG',
          utils.local_path()+'/data/processed_images/BUSHNELL_CROP_FULL.JPG',
          utils.local_path()+'/data/processed_images/MOULTRIE_M888_CROP_SHORT.JPG',
          utils.local_path()+'/data/processed_images/SPYPOINT_FORCE_CROP_SHORT.JPG',
          utils.local_path()+'/data/processed_images/BUSHNELL_CROP_SHORT.JPG']

for im_path in paths:
    config = ('-l eng --oem 1 --psm 3')
    im = cv2.imread(im_path,cv2.IMREAD_COLOR)
    text = pytesseract.image_to_string(im, config=config)
    print('tesseract english model output for %s is:\n%s\n'%(im_path.rsplit('/')[-1],text))

"""RESULTS FOR TESTS:
tesseract english model output for MOULTRIE_M888_CROP_FULL.JPG is:
at 2°c -MOULTRIECAM 02 APR 2018 11:17 am
tesseract english model output for SPYPOINT_FORCE_CROP_FULL.JPG is:
tesseract english model output for BUSHNELL_CROP_FULL.JPG is:
ATTICA 70F21 degree C at registeredtrademark 09-05-2018 07: 44: 23
tesseract english model output for MOULTRIE_M888_CROP_SHORT.JPG is:
2 degree C
MOULTRIECAM
02 APR 2018 11:17 am
tesseract english model output for SPYPOINT_FORCE_CROP_SHORT.JPG is:
05/18/201
04:51 pm
14: degreec
tesseract english model output for BUSHNELL_CROP_SHORT.JPG is:
10F21 degree C at 09-05-2018 O7: 44: 23
"""
