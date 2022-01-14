# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 00:16:24 2022

@author: Bianca Schröder
@project: GTSRB - Classification & Attacks using an CNN

"""


#%% Packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
class TrafficSignNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
# CONV => RELU => BN => POOL
        model.add(Conv2D(8, (5, 5), padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
# first set of (CONV => RELU => CONV => RELU) * 2 => POOL
        model.add(Conv2D(16, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
# second set of (CONV => RELU => CONV => RELU) * 2 => POOL
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
# first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
		# second set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
		# softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
		# return the constructed network architecture
        return model
    
#%% Packages    

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
#from pyimagesearch.trafficsignnet import TrafficSignNet
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


#%%

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
            # split the row into components and then grab the class ID
            # image path
        (label, imagePath) = row.strip().split(",")[-2:]
            # derive the full path to the image file and load it
        imagePath = os.path.sep.join([basePath, imagePath])
        image = io.imread(imagePath)
            # resize the image to be 32x32 pixels, ignoring aspect ratio,
            # and then perform Contrast Limited Adaptive Histogram
            # Equalization (CLAHE)
        image = transform.resize(image, (32, 32))
        image = exposure.equalize_adapthist(image, clip_limit=0.1)
            # update the list of data and labels, respectively
        data.append(image)
        labels.append(int(label))
        # convert the data and labels to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)
    # return a tuple of the data and labels
    return (data, labels)

#%% construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--C:/Users/Admin/Documents/Master Data Science/Semester 5/Deep Learning/project_1_deep_learning/data", required=True,
#	help="path to input GTSRB")
#ap.add_argument("-m", "--C:/Users/Admin/Documents/Master Data Science/Semester 5/Deep Learning/project_1_deep_learning/model2.py", required=True,
#	help="path to output model")
#ap.add_argument("-p", "--plot", type=str, default="plot.png",
#	help="path to training history plot")
args = vars(ap.parse_args())

#%% initialize the number of epochs to train for, base learning rate and batch size
NUM_EPOCHS = 15
INIT_LR = 1e-3
BS = 32
# load the label names
labelNames = open("signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]


#%%

# derive the path to the training and testing CSV files
trainPath = "Train.csv"
testPath = "Test.csv"
# load the training and testing data
print("[INFO] loading training and testing data...")
(trainX, trainY) = load_split("C:/Users/Admin/Documents/Master Data Science/Semester 5/Deep Learning/project_1_deep_learning/data",trainPath)
(testX, testY) = load_split("C:/Users/Admin/Documents/Master Data Science/Semester 5/Deep Learning/project_1_deep_learning/data",testPath)
# scale data to the range of [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0
# one-hot encode the training and testing labels
numLabels = len(np.unique(trainY))
trainY = to_categorical(trainY, numLabels)
testY = to_categorical(testY, numLabels)
# calculate the total number of images in each class and
# initialize a dictionary to store the class weights
classTotals = trainY.sum(axis=0)
classWeight = dict()
# loop over all classes and calculate the class weight
for i in range(0, len(classTotals)):
	classWeight[i] = classTotals.max() / classTotals[i]
    
    
#%% construct the image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	vertical_flip=False,
	fill_mode="nearest")
#%% initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
model = TrafficSignNet.build(width=32, height=32, depth=3, classes=numLabels)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
#%% train the network
print("[INFO] training network...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=trainX.shape[0] // BS,
	epochs=NUM_EPOCHS,
	class_weight=classWeight,
	verbose=1)

#%% evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=labelNames))
# save the network to disk
#print("[INFO] serializing network to '{}'...".format(args["model"]))
#model.save(args["model"])

#%%

from sklearn.metrics import accuracy_score
print(accuracy_score(testY.argmax(axis=1), predictions.argmax(axis=1)))

#%% Visualization


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Backgound style for the graphs
sns.set_style("darkgrid")

# Plot for the accuracy
plt.figure(0)
plt.plot(H.history['accuracy'], label='Training Accuracy', color="teal")
plt.plot(H.history['val_accuracy'], label='Validation Accuracy', color="magenta")
plt.title('Accuracy')  # title
plt.xlabel('Epochs')   # x-axis
plt.ylabel('Accuracy') # y-axis
plt.legend()           # key
plt.show()
# Plot for the loss
plt.figure(1)
plt.plot(H.history['loss'], label='Training Loss', color="teal")
plt.plot(H.history['val_loss'], label='Validation Loss', color="magenta")
plt.title('Loss')     # title
plt.xlabel('Epochs')  # x-axis
plt.ylabel('Loss')    # y-axis
plt.legend()          # key
plt.show()


#%% plot the training loss and accuracy
N = np.arange(0, NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
