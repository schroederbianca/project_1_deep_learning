# -*- coding: utf-8 -*-
"""
First approach at the neural network task from
https://www.kaggle.com/adarshangadi/recognizing-traffic-signs-using-convnets/notebook
"""


#%% Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
from PIL import Image
import os 
from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras import models, layers, losses

#%% Read the data

data = []
labels = []
classes = 43 
#cur_path = os.getcwd() 
file_path = "/Users/stoye/sciebo/Studium/39-Inf-DL - Deep Learning/projects/project_1_deep_learning/data"
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
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


#%% Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

#%% Build model

num_classes = 43

model = models.Sequential()

## Build Model
# 1st Conv layer 
model.add(layers.Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = X_train.shape[1:]))
model.add(layers.Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(layers.MaxPool2D(pool_size = (2, 2)))
# 2nd Conv layer        
model.add(layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(layers.MaxPool2D(pool_size = (2, 2)))
# Fully Connected layer        
model.add(layers.Flatten())
model.add(layers.Dense(256))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation="softmax"))


model.summary()

#%% Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#%% Train model

STEP_SIZE_TRAIN= 32
num_epochs = 15

history = model.fit(X_train, y_train,batch_size=STEP_SIZE_TRAIN,epochs=num_epochs,
                    validation_data=(X_test, y_test)) #, callbacks=[checkpoint])

#%% Test model

# here, wd should be the data folder
from sklearn.metrics import accuracy_score
y_test = pd.read_csv("/Users/stoye/sciebo/Studium/39-Inf-DL - Deep Learning/projects/project_1_deep_learning/data/Test.csv")
labels = y_test["ClassId"].values
imgs = y_test["Path"].values
data = []
for img in imgs:
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
X_test=np.array(data)
#pred = model.predict_classes(X_test) only works for tensorflow <= 2.5
pred = model.predict(X_test) 
pred = np.argmax(pred,axis=1)


#%% Accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred))

#%% Visualization

anc = history

# Relevant Packages
import matplotlib.pyplot as plt
import seaborn as sns

# Backgound style for the graphs
sns.set_style("darkgrid")

# Plot for the accuracy
plt.figure(0)
plt.plot(anc.history['accuracy'], label='Training Accuracy', color="teal")
plt.plot(anc.history['val_accuracy'], label='Validation Accuracy', color="magenta")
plt.title('Accuracy')  # title
plt.xlabel('Epochs')   # x-axis
plt.ylabel('Accuracy') # y-axis
plt.legend()           # key
plt.show()
# Plot for the loss
plt.figure(1)
plt.plot(anc.history['loss'], label='Training Loss', color="teal")
plt.plot(anc.history['val_loss'], label='Validation Loss', color="magenta")
plt.title('Loss')     # title
plt.xlabel('Epochs')  # x-axis
plt.ylabel('Loss')    # y-axis
plt.legend()          # key
plt.show()
