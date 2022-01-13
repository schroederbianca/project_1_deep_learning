# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 11:57:22 2022

@author: Bianca Schröder
@project: GTSRB - Classification & Attacks using an CNN

"""

#%% Install Requirements

#!pip install tensorflow 
#!pip install tensorflow keras 
#!pip install tensorflow sklearn 
#!pip install tensorflow matplotlib 
#!pip install tensorflow pandas 
#!pip install tensorflow pil


#%% Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
from PIL import Image
import os 
from sklearn.model_selection import train_test_split 
from keras.utils import to_categorical 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

#%% Read the data

data = []
labels = []
classes = 43 
#cur_path = os.getcwd() 
file_path = "C:/Users/Admin/Documents/Master Data Science/Semester 5/Deep Learning/project_1_deep_learning/data"
for i in range(classes): 
    path = os. path.join(file_path,'train', str(i)) 
    images = os.listdir(path) 
    for a in images: 
        try: 
            image = Image.open(path + '/' + a) 
            image = image.resize((30,30)) 
            image = np.array(image) 
            data.append(image) 
            labels.append(i) 
        except: 
            print("Error loading image") 
            
            
data = np.array(data)
labels = np.array(labels)

#%% Data Shape
print(data.shape, labels.shape)


#%% Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_t1.shape, X_t2.shape, y_t1.shape, y_t2.shape)


#%% Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

#%% Building the Model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#%% Model Training 

eps = 15
anc = model.fit(X_train, y_train, batch_size=32, epochs=eps, validation_data=(X_test, y_test))

#%% Plotting graphs for Accuracy
plt.figure(0)
plt.plot(anc.history['accuracy'], label='Training Accuracy')
plt.plot(anc.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(anc.history['loss'], label='Training Loss')
plt.plot(anc.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%% Nice plot
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_style("darkgrid")
plt.figure(0)
plt.plot(anc.history['accuracy'], label='Training Accuracy', color="teal")
plt.plot(anc.history['val_accuracy'], label='Validation Accuracy', color="magenta")
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(anc.history['loss'], label='Training Loss', color="teal")
plt.plot(anc.history['val_loss'], label='Validation Loss', color="magenta")
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#%% Model Testing 

from sklearn.metrics import accuracy_score
y_test = pd.read_csv("C:/Users/Admin/Documents/Master Data Science/Semester 5/Deep Learning/project_1_deep_learning/data/Test.csv")
labels = y_test["ClassId"].values
imgs = y_test["Path"].values
data=[]
for img in imgs:
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
X_test=np.array(data)
pred = model.predict_classes(X_test)


#%% Accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred))

#%% Save the Model

model.save(‘traffic_classifier.h5’)






