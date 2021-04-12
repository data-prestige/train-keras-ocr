'''
Created on 18/02/2021

@author: Francesco Pugliese
'''

#from View.view import View 

import pandas as pd
import glob
import cv2

# import os
import os
from os.path import isfile, isdir, join
import pdb

#other imports
import numpy as np
import timeit
from imutils import paths

class Preprocessing(object):
    def __init__(self, params):
        self.__params = params
    
    def load_license_plate_detection_data_from_local_disk(image_path, resize=False, verbose=False):
        
        if verbose == True:
            print ("\nLoading data from the Local Disk...")

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        if resize:
            img = cv2.resize(img, (224,224))
        return img
