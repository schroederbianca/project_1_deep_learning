# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 11:57:22 2022

@author: Bianca Schröder & Ferdinand Stoye
@project: GTSRB - Classification & Attacks using an CNN

"""

#%% Install Requirements

#!pip install tensorflow 
#!pip install tensorflow keras 
#!pip install tensorflow sklearn 
#!pip install tensorflow matplotlib 
#!pip install tensorflow pandas 
#!pip install tensorflow pil
#!pip install adversarial-robustness-toolbox



#%% Packages

# analysis
import numpy as np
import pandas as pd
import os 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

# tensorflow
import tensorflow as tf 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
if tf.__version__[0]!="2": # check tensorflow version
    raise ImportError("This notebook requires Tensofrlow v2.")

# art
from art.attacks.evasion import FastGradientMethod, PixelAttack
from art.estimators.classification import KerasClassifier

# plots
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns



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

# Data shape
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
model.summary()

#%% Create the ART classifier
tf.compat.v1.disable_eager_execution() # to make the classifier work
classifier = KerasClassifier(model=model, clip_values=(0,30))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#%% Train the ART classifier
history = classifier.fit(X_train, y_train, nb_epochs=10)

#%% Evaluate performance for clean data
y_test = pd.read_csv("/Users/stoye/sciebo/Studium/39-Inf-DL - Deep Learning/projects/project_1_deep_learning/data/Test.csv")
labels = y_test["ClassId"].values
imgs = y_test["Path"].values
data=[]
for img in imgs:
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
X_test=np.array(data)

pred = model.predict(X_test)
pred = np.argmax(pred,axis=1)
print(accuracy_score(labels, pred))

#%% Create attacked data
attack = FastGradientMethod(estimator=classifier, eps=5, eps_step=2)
X_test = X_test.astype('float32')
x_test_adv = attack.generate(x=X_test)

#%% Evaluate performance for attacked data
predictions = classifier.predict(x_test_adv)
predictions = np.argmax(predictions,axis=1)
accuracy_test = accuracy_score(labels, predictions)
perturbation = np.mean(np.abs((x_test_adv - X_test)))
print('Accuracy on adversarial test data: {:4.2f}%'.format(accuracy_test * 100))
print('Average perturbation: {:4.2f}'.format(perturbation))

#%% Visualize one attacked image
plt.matshow(x_test_adv[0])
plt.show()

plt.matshow(X_test[0])
plt.show()

#%% Visualize the performance

sns.set_style("darkgrid")
plt.figure(0)
plt.plot(history.history['accuracy'], label='Training Accuracy', color="teal")
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color="magenta")
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(history.history['loss'], label='Training Loss', color="teal")
plt.plot(history.history['val_loss'], label='Validation Loss', color="magenta")
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



#%% Show one picture of the data
image_42 = Image.open(y_test["Path"].values[42])
image_42.show()

#%%

# Step 3: Create the ART classifier

#%%


#%% Step 4: Train the ART classifier -> clean model

classifier.fit(X_train, y_train, nb_epochs=3)

#%% Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(X_test)
predictions = np.argmax(predictions,axis=1)
print(accuracy_score(labels, predictions))


#accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
#print("Accuracy on benign test examples: {}%".format(accuracy * 100))

#%% Step 6: Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=5, eps_step=2)
x_test_adv = attack.generate(x=X_test)

#%% Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
predictions = np.argmax(predictions,axis=1)
print(accuracy_score(labels, predictions))
#accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
#print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))





# THE END
#####
#####
#####

#%% Accuracy with the test data
print(accuracy_score(labels, pred))

#%% Save the Model

#model.save("traffic_classifier.h5")



#%% 
X_train = X_train.astype('float32')

#%%








#%%
#%%
attack = PixelAttack(classifier=classifier)






#%%

# OK, working order:
    # 1. Build model
    # 2. Compile model
    # 3. Create Keras classifier from art
    # 4. Train model (with Keras classifier.fit function)


