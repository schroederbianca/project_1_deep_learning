# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 16:41:18 2022

@author: Bianca Schröder
@project: GTSRB - Classification & Attacks using an CNN
@content: This file is used for visualizing the results of the training epochs of the specific models.
    
Attention: The fitted model has to be saved as anc!
    
"""
# Relevant Packages
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
