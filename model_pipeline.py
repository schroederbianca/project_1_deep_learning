# -*- coding: utf-8 -*-
"""
@author: Bianca Schröder & Ferdinand Stoye
@project: GTSRB - Classification & Attacks using an CNN & different attacks from ART

This file contains all main functionalities of the GTSRB project in the 
module 39-DL Deep Learning (summer term 2021)

Content: 
    - functions to create three CNNs to classify the data set
    - function to train a previously compiled model
    - function to attack a trained model, using one of five specified attacks
    - function to plot an instance (image) of the data set
    - function to compare the class predictions of the previously attacked models
      for a specified image
    - function to plot an image and its attacked versions
"""



#%% Model creation functions

def create_model_1(verbose=True):
    '''
    :param verbose: Determines if model summary is printed
    
    :return model: Compiled CNN
    
    Creates and compiles a CNN, referred to as CNN1.
    '''
    
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
    # Compilation of the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if verbose:
        print(model.summary())
    return model
    
def create_model_2(verbose=True):
    '''
    :param verbose: Determines if model summary is printed
    
    :return model: Compiled CNN
    
    Creates and compiles a CNN, referred to as CNN2.
    '''
    
    model = Sequential()
    chanDim = -1
    # CONV => RELU => BN => POOL
    model.add(Conv2D(8, (5, 5), padding="same",input_shape=X_train.shape[1:]))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # first set of (CONV => RELU => CONV => RELU) * 2 => POOL
    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # second set of (CONV => RELU => CONV => RELU) * 2 => POOL
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # first set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(128))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dropout(0.5))
    # second set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(128))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dropout(0.5))
    # softmax classifier
    model.add(Dense(classes))
    model.add(tf.keras.layers.Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    if verbose:
        print(model.summary())
    return model
    
def create_model_3(verbose=True):
    '''
    :param verbose: Determines if model summary is printed
    
    :return model: Compiled CNN
    
    Creates and compiles a CNN, referred to as CNN3.
    '''
    
    model = tf.keras.models.Sequential()
    ## Build Model
    # 1st Conv layer 
    model.add(tf.keras.layers.Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = X_train.shape[1:]))
    model.add(tf.keras.layers.Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(tf.keras.layers.MaxPool2D(pool_size = (2, 2)))
    # 2nd Conv layer        
    model.add(tf.keras.layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(tf.keras.layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(tf.keras.layers.MaxPool2D(pool_size = (2, 2)))
    # Fully Connected layer        
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(43, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if verbose:
        print(model.summary())
    return model


#%% Model training function

def train_model(model, X_train, X_test, y_train, y_test, labels, epochs=15, verbose=True):
    '''
    :param model: Determines which model should be read in, one of ['model_1', 'model_2', 'model_3']
    :param X_train: Training features
    :param X_test: Test features
    :param y_train: Training labels
    :param y_test: Test labels
    :param labels: Additional labels
    :param epochs: Number of epochs to train
    :param verbose: Determines if training time is reported
    
    :return classifier: Trained Keras classifier
    
    Trains the compiled CNN, tests its accuracy and returns the classifier.
    '''
    # Read in model specified in function call
    if model == 'model_1':
        model = create_model_1()
    elif model == 'model_2':
        model = create_model_2()
    elif model == 'model_3':
        model = create_model_3()
    else:
        return "Error: Please choose from ['model_1, 'model_2', 'model_3']"
    
    # Create Keras classifier
    classifier = KerasClassifier(model=model, clip_values=(0,30))
    start = time.time()
    # Train model, ensure that we have floating point numbers
    # Batch size could be increased in theory, depending on available memory
    history = classifier.fit(X_train.astype('float32'), y_train, nb_epochs=epochs, batch_size=32)
    end = time.time()
    if verbose:
        print("Training time: {0}".format(end-start))

    # Predict test data labels
    pred = model.predict(X_test)
    pred = np.argmax(pred,axis=1)
    # Report test accuracy
    print(f"Accuracy for clean data: {accuracy_score(labels, pred)}")
    
    return classifier


#%% Attack function

def attack_model(classifier, attack, X_test, verbose=True): 
    '''
    :param classifier: Trained Keras classifier
    :param attack: The attack to perform, one of ['fgm', 'few_pixel_opt', 'few_pixel_rand', 'backdoor_poison', 'universal_perturbation']
    :param X_test: Test features
    :param verbose: Determines if attacking time is reported
    
    :return list : List of important measurements. In the case of the randomized
                   few pixel attack containing accuracy on test data, 
                   average perturbation and attacked data. Otherwise containing
                   accuracy on test data, average perturbation, attacked data and
                   the created attack object by ART
    
    Attacks the trained CNN, tests its accuracy and returns the classifier.
    '''
    
    # Read in test labels
    y_test = pd.read_csv("/Users/stoye/sciebo/Studium/39-Inf-DL - Deep Learning/projects/project_1_deep_learning/data/Test.csv")
    labels = y_test["ClassId"].values
    
    # Determine which attack to use and perform attack
    if attack == 'fgm':
        start = time.time()
        # Fast Gradient method
        attack_obj = FastGradientMethod(estimator=classifier, eps=7, eps_step=3)
        attacked_data = attack_obj.generate(x=X_test.astype('float32'))
    elif attack == 'few_pixel_opt':
        # constrained to 1000 data points, due to runtime. Could be extended in theory
        n_examples = 1000
        start = time.time()
        # (Optimized) pew pixel attack
        attack_obj = PixelAttack(classifier=classifier, th=4)#, th=10)
        attacked_data = attack_obj.generate(x=X_test[0:n_examples].astype(int), max_iter=5)
    elif attack == 'few_pixel_rand':
        start = time.time()
        file_path = "/Users/stoye/sciebo/Studium/39-Inf-DL - Deep Learning/projects/project_1_deep_learning/data"
        imgs = y_test["Path"].values
        #cur_path = os.getcwd() 
        image_data=[]
        th = 4
        # for each image, place th pixels at a random position in a random color
        for img in imgs:
            image = Image.open(file_path+"/"+img)
            image = image.resize((30,30))
            for i in range(th):
                position = tuple(np.random.choice(range(30), size=2))
                new_pixels = tuple(np.random.choice(range(256), size=3))
                image.putpixel((position),(new_pixels))
            image_data.append(np.array(image))
        attacked_data = np.array(image_data)
    elif attack == 'backdoor_poison':
        start = time.time()
        # Backdoor poisoning attack
        # Trigger image is taken arbitrarily: 'Train/8/00008_00000_00014.png'
        # (Manipulation idea: classify street signs wrongly as 120kmh signs, to provoke accidents)
        attack_obj = PoisoningAttackBackdoor(lambda x: insert_image(x, 
                    backdoor_path='Train/8/00008_00000_00014.png', size=(10,10),
                    mode='RGB', blend=0.8, random=True))
        attacked_data, poisoned_y = attack_obj.poison(X_test.astype('float32'), labels)
    elif attack == 'universal_perturbation':
        start = time.time()
        # Universal perturbation attack using Fast Gradient sign method
        attack_obj = UniversalPerturbation(classifier, attacker='fgsm', eps=10, max_iter=10,
                                                    norm='inf', delta=0.4, batch_size=128)
        attacked_data = attack_obj.generate(x=X_test/255, max_iter=5)#, y=labels)
    else:
        return "Error: Please choose from ['fgm, 'few_pixel_opt', 'few_pixel_rand', 'backdoor_poison', 'universal_perturbation']"

    # Ensure floating point numbers
    X_test = X_test.astype('float32')
    
    # Evaluate performance for attacked data
    if attack == 'few_pixel_opt':
        # in this case we have only 1000 images due to runtime constraints
        predictions = classifier.predict(attacked_data)#.astype('float32'))
        predictions = np.argmax(predictions,axis=1)
        accuracy_test = accuracy_score(labels[0:n_examples], predictions)
        perturbation = np.mean(np.abs((attacked_data - X_test[0:n_examples])))
        #print('Accuracy on adversarial test data: {:4.5f}%'.format(accuracy_test * 100))
        #print('Average perturbation: {:4.5f}'.format(perturbation))
        end = time.time()
        
    else:
        predictions = classifier.predict(attacked_data.astype('float32'))
        predictions = np.argmax(predictions,axis=1)
        accuracy_test = accuracy_score(labels, predictions)
        perturbation = np.mean(np.abs((attacked_data - X_test)))
        #print('Accuracy on adversarial test data: {:4.5f}%'.format(accuracy_test * 100))
        #print('Average perturbation: {:4.5f}'.format(perturbation))
        end = time.time()
    if verbose:
        print("Overall attack time: {0}".format(end-start))

    # in the few_pixel_rand attack we do not have an attack_obj
    if attack == 'few_pixel_rand':
        return [accuracy_test, perturbation, attacked_data]
    else:
        return [accuracy_test, perturbation, attacked_data, attack_obj]



#%% Plot image

def plot_image(data, image_number, mode='int'):
    '''
    :param data: Trained Keras classifier
    :param image_number: Image number in the provided data
    :param mode: How is the rgb color coding? 'int' -- 0-255, 'float' -- 0-1
    
    Plots a specified image from provided data.
    '''
    
    if mode == 'int':
        plt.imshow(data[image_number].squeeze().astype(int))
    if mode == 'float':
        plt.imshow(data[image_number].squeeze()/255)




#%% Compare class predictions

def compare_class_predictions(image_number, nb_classes=3):
        '''
        :param image_number: Image number in the test data
        :param nb_classes: The nb_classes most likely classes are reported
        
        Predicts the attacked data and reportes the most likely classes for a 
        specified image based on model_1. 
        Note: the Classifier objects and models have to be prepared exactly as 
        used in this function:
            First model: model_1
            FGM attack: attack_m1_fgm
            Few pixel optimized attack: attack_m1_fpo
            Few pixel randomized attack: attack_m1_fpr
            Backdoor poisoning attack: attack_m1_bp
            Universal perturbation attack: attack_m1_up
        '''
        
        print(f"Processing image {image_number}...")
        # original prediction
        predicted = model_1.predict(X_test.astype('float32'))
        predicted_class_orig = predicted[image_number].argsort()[-nb_classes:][::-1]
        print(f"Most likely classes using original test data: {predicted_class_orig}")
    
        # predicted class for this image -> attacked with Fast Gradient
        predicted_FG = model_1.predict(attack_m1_fgm[2].astype('float32'))
        predicted_class_FG = predicted_FG[image_number].argsort()[-nb_classes:][::-1]
        print(f"Most likely classes using Fast Gradient test data: {predicted_class_FG}")
    
        # predicted class for this image -> attacked with Few Pixel
        predicted_FPO = model_1.predict(attack_m1_fpo[2].astype('float32'))
        predicted_class_FPO = predicted_FPO[image_number].argsort()[-nb_classes:][::-1]
        print(f"Most likely classes using optimized Few Pixel test data: {predicted_class_FPO}")
        
        # predicted class for this image -> attacked with Few Pixel
        predicted_FPR = model_1.predict(attack_m1_fpr[2].astype('float32'))
        predicted_class_FPR = predicted_FPR[image_number].argsort()[-nb_classes:][::-1]
        print(f"Most likely classes using randomized Few Pixel test data: {predicted_class_FPR}")
    
        # predicted class for this image -> attacked with Backdoor Poisoning
        predicted_BP = model_1.predict(attack_m1_bp[2].astype('float32'))
        predicted_class_BP = predicted_BP[image_number].argsort()[-nb_classes:][::-1]
        print(f"Most likely classes using Backdoor Poisoning test data: {predicted_class_BP}")

        # predicted class for this image -> attacked with Universal Perturbation
        predicted_UP = model_1.predict(attack_m1_up[2].astype('float32'))
        predicted_class_UP = predicted_UP[image_number].argsort()[-nb_classes:][::-1]
        print(f"Most likely classes using Universal Perturbation test data: {predicted_class_UP}")



#%% Plot an image and its attacked versions (from test set)
def plot_image_versions(image_number):
    '''
    :param image_number: Image number in the test data
    
    Plots an image from the test data and all its attacked versions.
    Note: the Classifier objects have to be prepared exactly as 
    used in this function:
        FGM attack: attack_m1_fgm
        Few pixel optimized attack: attack_m1_fpo
        Few pixel randomized attack: attack_m1_fpr
        Backdoor poisoning attack: attack_m1_bp
        Universal perturbation attack: attack_m1_up
    '''
    
    # original
    plt.imshow(X_test[image_number].squeeze().astype(int))
    plt.show()
    # fast gradient
    plt.imshow(attack_m1_fgm[2][image_number].squeeze().astype(int))
    plt.show()
    # few pixel optimized
    plt.imshow(attack_m1_fpo[2][image_number].squeeze().astype(int))
    plt.show()
    # few pixel randomized
    plt.imshow(attack_m1_fpr[2][image_number].squeeze().astype(int))
    plt.show()
    # backdoor poisoning
    plt.imshow(attack_m1_bp[2][image_number].squeeze())
    plt.show()
    # universal perturbation
    plt.imshow(attack_m1_up[2][image_number].squeeze())
    plt.show()













#%% Packages

# analysis
import numpy as np
import pandas as pd
import os 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import time

# tensorflow
import tensorflow as tf 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
if tf.__version__[0]!="2": # check tensorflow version
    raise ImportError("This notebook requires Tensofrlow v2.")

# art
from art.attacks.evasion import FastGradientMethod, PixelAttack, UniversalPerturbation
from art.estimators.classification import KerasClassifier
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import insert_image

# plots
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns


#%% Read the data

data = []
labels = []
classes = 43 
#file_path = "/Users/stoye/sciebo/Studium/39-Inf-DL - Deep Learning/projects/project_1_deep_learning/data"
# the current working directory should be the location of the file
file_path = os.getcwd() 
# Read each image file and store data and labels
for i in range(classes): 
    path = os.path.join(file_path,'data/train', str(i)) 
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

#%% Load correct test data, preprocessing
y_test = pd.read_cev(os.path.join(os.getcwd(),'data/Test.csv'))
#y_test = pd.read_csv("/Users/stoye/sciebo/Studium/39-Inf-DL - Deep Learning/projects/project_1_deep_learning/data/Test.csv")
labels = y_test["ClassId"].values
imgs = y_test["Path"].values
data=[]
# Resize all images to 30x30 to ensure comparability
for img in imgs:
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
X_test=np.array(data)

#%% Create and train model
tf.compat.v1.disable_eager_execution() # to make the classifier work, has to be executed before the model is built

# Trained model objects
model_1 = train_model('model_1', X_train, X_test, y_train, y_test, labels, epochs=15, verbose=True)
model_2 = train_model('model_2', X_train, X_test, y_train, y_test, labels, epochs=15, verbose=True)
model_3 = train_model('model_3', X_train, X_test, y_train, y_test, labels, epochs=15, verbose=True)

#%% Create and evaluate attacks for each model

# Fast Gradient method
attack_m1_fgm = attack_model(model_1, 'fgm', X_test)
print("Attack model_1 with Fast Gradient method:")
print(f"Accuracy on attacked data: {attack_m1_fgm[0]}")
print(f"Average perturbation on attacked data: {attack_m1_fgm[1]}")

attack_m2_fgm = attack_model(model_2, 'fgm', X_test)
print("Attack model_2 with Fast Gradient method:")
print(f"Accuracy on attacked data: {attack_m2_fgm[0]}")
print(f"Average perturbation on attacked data: {attack_m2_fgm[1]}")

attack_m3_fgm = attack_model(model_3, 'fgm', X_test)
print("Attack model_3 with Fast Gradient method:")
print(f"Accuracy on attacked data: {attack_m3_fgm[0]}")
print(f"Average perturbation on attacked data: {attack_m3_fgm[1]}")

#%%

# Few Pixel attack optimized
attack_m1_fpo = attack_model(model_1, 'few_pixel_opt', X_test)
print("Attack model_1 with few pixel attack (optimized):")
print(f"Accuracy on attacked data: {attack_m1_fpo[0]}")
print(f"Average perturbation on attacked data: {attack_m1_fpo[1]}")

attack_m2_fpo = attack_model(model_2, 'few_pixel_opt', X_test)
print("Attack model_2 with few pixel attack (optimized):")
print(f"Accuracy on attacked data: {attack_m2_fpo[0]}")
print(f"Average perturbation on attacked data: {attack_m2_fpo[1]}")

attack_m3_fpo = attack_model(model_3, 'few_pixel_opt', X_test)
print("Attack model_3 with few pixel attack (optimized):")
print(f"Accuracy on attacked data: {attack_m3_fpo[0]}")
print(f"Average perturbation on attacked data: {attack_m3_fpo[1]}")

#%%

# Few Pixel attack randomized
attack_m1_fpr = attack_model(model_1, 'few_pixel_rand', X_test)
print("Attack model_1 with few pixel attack (randomized):")
print(f"Accuracy on attacked data: {attack_m1_fpr[0]}")
print(f"Average perturbation on attacked data: {attack_m1_fpr[1]}")

attack_m2_fpr = attack_model(model_2, 'few_pixel_rand', X_test)
print("Attack model_2 with few pixel attack (randomized):")
print(f"Accuracy on attacked data: {attack_m2_fpr[0]}")
print(f"Average perturbation on attacked data: {attack_m2_fpr[1]}")

attack_m3_fpr = attack_model(model_3, 'few_pixel_rand', X_test)
print("Attack model_3 with few pixel attack (randomized):")
print(f"Accuracy on attacked data: {attack_m3_fpr[0]}")
print(f"Average perturbation on attacked data: {attack_m3_fpr[1]}")

#%%

# Backdoor Poisoning
attack_m1_bp = attack_model(model_1, 'backdoor_poison', X_test)
print("Attack model_1 with backdoor poisoning attack:")
print(f"Accuracy on attacked data: {attack_m1_bp[0]}")
print(f"Average perturbation on attacked data: {attack_m1_bp[1]}")

attack_m2_bp = attack_model(model_2, 'backdoor_poison', X_test)
print("Attack model_2 with backdoor poisoning attack:")
print(f"Accuracy on attacked data: {attack_m2_bp[0]}")
print(f"Average perturbation on attacked data: {attack_m2_bp[1]}")

attack_m3_bp = attack_model(model_3, 'backdoor_poison', X_test)
print("Attack model_3 with backdoor poisoning attack:")
print(f"Accuracy on attacked data: {attack_m3_bp[0]}")
print(f"Average perturbation on attacked data: {attack_m3_bp[1]}")

#%%

# Universal Perturbation
attack_m1_up = attack_model(model_1, 'universal_perturbation', X_test)
print("Attack model_1 with Universal Perturbation attack:")
print(f"Accuracy on attacked data: {attack_m1_up[0]}")
print(f"Average perturbation on attacked data: {attack_m1_up[1]}")

attack_m2_up = attack_model(model_2, 'universal_perturbation', X_test)
print("Attack model_2 with Universal Perturbation attack:")
print(f"Accuracy on attacked data: {attack_m2_up[0]}")
print(f"Average perturbation on attacked data: {attack_m2_up[1]}")

attack_m3_up = attack_model(model_3, 'universal_perturbation', X_test)
print("Attack model_3 with Universal Perturbation attack:")
print(f"Accuracy on attacked data: {attack_m3_up[0]}")
print(f"Average perturbation on attacked data: {attack_m3_up[1]}")

#%% Class predictions

## Compare predictions for image 1
compare_class_predictions(image_number=1)


#%% Image versions

## Print image 1
plot_image_versions(image_number=1)




