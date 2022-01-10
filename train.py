# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 17:52:31 2022

@author: Bianca Schröder
"""
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from pyimagesearch.trafficsignnet import TrafficSignNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from skimage import transform
from skimage import exposure
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import os


def load_split(basePath, csvPath):
	# initialize the list of data and labels
	data = []
	labels = []
	# load the contents of the CSV file, remove the first line (since
	# it contains the CSV header), and shuffle the rows (otherwise
	# all examples of a particular class will be in sequential order)
	rows = open(csvPath).read().strip().split("\n")[1:]
	random.shuffle(rows)
    
    
# loop over the rows of the CSV file
for (i, row) in enumerate(rows):
# check to see if we should show a status update
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {} total images".format(i))
        # split the row into components and then grab the class ID and image path
        (label, imagePath) = row.strip().split(",")[-2:]
        # derive the full path to the image file and load it
		imagePath = os.path.sep.join([basePath, imagePath])
		image = io.imread(imagePath)
        
        
        