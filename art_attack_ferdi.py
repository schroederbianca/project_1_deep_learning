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
import time

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
# Here we should source the model_x file for the creation of the different models
tf.compat.v1.disable_eager_execution() # to make the classifier work, has to be executed before the model is built
# This is model_1:
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

#%% plot model architecture
#from tensorflow.keras.utils import plot_model
#plot_model(model, to_file='model1.png', rankdir='LR')

#%% Create the ART classifier 
classifier = KerasClassifier(model=model, clip_values=(0,30))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#%% Train the ART classifier
start = time.time()
history = classifier.fit(X_train, y_train, nb_epochs=15, batch_size=32)
end = time.time()
print("Training time: {0}".format(end-start))
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

#%% Attack 1: Fast Gradient Method
attack_fast_gradient = FastGradientMethod(estimator=classifier, eps=7, eps_step=3)
X_test = X_test.astype('float32')
x_test_adv_fast_gradient = attack_fast_gradient.generate(x=X_test)

# Evaluate performance for attacked data
predictions = classifier.predict(x_test_adv_fast_gradient)
predictions = np.argmax(predictions,axis=1)
accuracy_test = accuracy_score(labels, predictions)
perturbation = np.mean(np.abs((x_test_adv_fast_gradient - X_test)))
print('Accuracy on adversarial test data: {:4.5f}%'.format(accuracy_test * 100))
print('Average perturbation: {:4.5f}'.format(perturbation))

#%% plot a picture and its attacked version
plt.imshow(X_test[42].squeeze().astype(int))

plt.imshow(x_test_adv_fast_gradient[42].squeeze().astype(int))

#%% Attack 2: Pixel Attack -> takes forever (nach 1 Stunde abgebrochen)
n_examples = 1000
attack_few_pixel = PixelAttack(classifier=classifier, th=4)#, th=10)
x_test_adv_few_pixel = attack_few_pixel.generate(x=X_test[0:n_examples].astype(int), max_iter=5)

# Evaluate performance for attacked data
predictions = classifier.predict(x_test_adv_few_pixel)
predictions = np.argmax(predictions,axis=1)
accuracy_test = accuracy_score(labels[0:n_examples], predictions)
perturbation = np.mean(np.abs((x_test_adv_few_pixel[0:n_examples] - X_test[0:n_examples])))
print('Accuracy on adversarial test data: {:4.5f}%'.format(accuracy_test * 100))
print('Average perturbation: {:4.5f}'.format(perturbation))

#%% plot a picture and its attacked version
#plt.imshow(X_test[42].squeeze().astype(int))

plt.imshow(x_test_adv_few_pixel[42].squeeze().astype(int))


#%% Attack 3: Backdoor Attack: Data poisoning
# Code adapted from: https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_attack_backdoor_image.ipynb
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import insert_image


# We have to declare an image that we want to be backdoored.
# We want to manipulate pictures such that the 120 sign gets detected, so we take
# data/Train/8/00008_00000_00014.png as the backdoor image
attack_backdoor_poisoning = PoisoningAttackBackdoor(lambda x: insert_image(x, 
                                                                 backdoor_path='Train/8/00008_00000_00014.png',
                                                                 size=(10,10),
                                                                 mode='RGB', blend=0.8, random=True
                                                                ))
poisoned_x, poisoned_y = attack_backdoor_poisoning.poison(X_test, labels)

# Evaluate performance for attacked data
predictions = classifier.predict(poisoned_x)
predictions = np.argmax(predictions,axis=1)
accuracy_test = accuracy_score(labels, predictions)
perturbation = np.mean(np.abs((poisoned_x - X_test)))
print('Accuracy on adversarial test data: {:4.5f}%'.format(accuracy_test * 100))
print('Average perturbation: {:4.2f}'.format(perturbation))

#%% Visualize one attacked image with the backdoor attack
plt.imshow(poisoned_x[42].squeeze())


#%% Attack 4: Alternative Backdoor Attack: Clean Label Backdoor Attack
# Code adapted from https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_attack_clean_label_backdoor.ipynb
from art.attacks.poisoning import PoisoningAttackBackdoor, PoisoningAttackCleanLabelBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from art.utils import to_categorical
from art.estimators.classification import KerasClassifier
from art.defences.trainer import AdversarialTrainerMadryPGD


# Poison training data
percent_poison = 0.33
# Shuffle training data
n_train = np.shape(y_train)[0]
shuffled_indices = np.arange(n_train)
np.random.shuffle(shuffled_indices)
X_train_shuffled = X_train[shuffled_indices]
y_train_shuffled = y_train[shuffled_indices]

# Now we poison the data, this happens BEFORE the model gets trained
backdoor = PoisoningAttackBackdoor(add_pattern_bd)
example_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, # we want class 8
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0])
pdata, plabels = backdoor.poison(X_test, y=example_target)

plt.imshow(pdata[42].squeeze().astype(int))

# Poison some percentage of all non-class 8 to class 8
targets = to_categorical([8], 43)[0] 
model_clean_label_backdoor = AdversarialTrainerMadryPGD(classifier, nb_epochs=15, eps=0.15, eps_step=0.001)
model_clean_label_backdoor.fit(X_train_shuffled, y_train_shuffled)

attack_clean_label_backdoor = PoisoningAttackCleanLabelBackdoor(backdoor=backdoor, 
                                           proxy_classifier=model_clean_label_backdoor.get_classifier(),
                                           target=targets, pp_poison=percent_poison, norm=2, eps=5,
                                           eps_step=0.1, max_iter=200)
pdata, plabels = attack_clean_label_backdoor.poison(X_train_shuffled, y_train_shuffled)
# This seems to take some time





#%% Compare predictions of image 42

def compare_class_predictions(image_number, nb_classes=1):
    print(f"Processing image {image_number}...")
    # original prediction
    predicted = classifier.predict(X_test)
    #predicted_class_orig = np.argmax(predicted[image_number])
    predicted_class_orig = predicted[image_number].argsort()[-nb_classes:][::-1]
    print(f"Most likely classes using original test data: {predicted_class_orig}")

    # predicted class for this image -> attacked with Fast Gradient
    predicted_FG = classifier.predict(x_test_adv_fast_gradient)
    #predicted_class_FG = np.argmax(predicted_FG[image_number])
    predicted_class_FG = predicted_FG[image_number].argsort()[-nb_classes:][::-1]
    print(f"Most likely classes using Fast Gradient test data: {predicted_class_FG}")

    # predicted class for this image -> attacked with Few Pixel
    predicted_FP = classifier.predict(x_test_adv_few_pixel)
    #predicted_class_FG = np.argmax(predicted_FG[image_number])
    predicted_class_FP = predicted_FP[image_number].argsort()[-nb_classes:][::-1]
    print(f"Most likely classes using Few Pixel test data: {predicted_class_FP}")
    
    # predicted class for this image -> attacked with Backdoor Poisoning
    predicted_BP = classifier.predict(poisoned_x)
    #predicted_class_BP = np.argmax(predicted_BP[image_number])
    predicted_class_BP = predicted_BP[image_number].argsort()[-nb_classes:][::-1]

    print(f"Most likely classes using Backdoor Poisoning test data: {predicted_class_BP}")
    

compare_class_predictions(42, 3)
# for image 2002 there are completly different results!




#%% Plot an image and its attacked versions (from test set)
def plot_image_versions(image_number):
    # original
    plt.imshow(X_test[image_number].squeeze().astype(int))
    plt.show()
    # fast gradient
    plt.imshow(x_test_adv_fast_gradient[image_number].squeeze().astype(int))
    plt.show()
    # few pixel
    plt.imshow(x_test_adv_few_pixel[image_number].squeeze().astype(int))
    plt.show()
    # backdoor poisoning
    plt.imshow(poisoned_x[image_number].squeeze())
    plt.show()


#%%
plot_image_versions(666)











#%% Visualize the performance, doesn't work yet

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




# THE END
#####
#####
#####







#%%
#%%






#%%

# OK, working order:
    # 1. Build model
    # 2. Compile model
    # 3. Create Keras classifier from art
    # 4. Train model (with Keras classifier.fit function)



