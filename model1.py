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
import PIL
from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

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
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


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

eps = 3
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
#pred = model.predict_classes(X_test)
# Da Zeile 146 ab Tensorflow 2.6 nicht mehr funktioniert
pred = model.predict(X_test)
pred = np.argmax(pred,axis=1)

print(accuracy_score(labels, pred))

#%% Print pic
#print(pred)
#print(Image.open(y_test["Path"].values[42]))
image_42 = Image.open(y_test["Path"].values[42])
image_42.show()


#%% Test attack
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
#tf.compat.v1.disable_eager_execution()
#%%
tf.compat.v1.disable_eager_execution()
#%%
if tf.__version__[0]!="2":
    raise ImportError("Tjis notebook requires Tensofrlow v2.")

#%%

# Step 3: Create the ART classifier

classifier = KerasClassifier(model=model, clip_values=(0,30))
#%%
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#%% Step 4: Train the ART classifier

classifier.fit(X_train, y_train, nb_epochs=eps)

#%% Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(X_test)
predictions = np.argmax(predictions,axis=1)
print(accuracy_score(labels, predictions))


#accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
#print("Accuracy on benign test examples: {}%".format(accuracy * 100))

#%% Step 6: Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=X_test)

#%% Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
predictions = np.argmax(predictions,axis=1)
print(accuracy_score(labels, predictions))
#accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
#print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))








#%% Accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred))

#%% Save the Model

#model.save("traffic_classifier.h5")



#%% 
from art.estimators.classification import KerasClassifier
X_train = X_train.astype('float32')

#%%








#%% Step 3: Create the ART classifier

classifier = KerasClassifier(model=model)#, use_logits=False, clip_values=(0, 32))

#%% Step 4: Train the ART classifier

classifier.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test), nb_epochs=15)

#%% Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(X_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

#%% Step 6: Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=X_test)

#%% Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))


#%%
from art.attacks.evasion import PixelAttack
#%%
attack = PixelAttack(classifier=classifier)
