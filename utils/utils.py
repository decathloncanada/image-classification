# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:28:41 2018

Utils functions for the image-classifier

split_train: function to build a validation set from a training set of images.

@author: AI team
"""

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

from random import shuffle

def split_train(path=parentdir+'\\data\\image_dataset', split=0.1):
    """
    path: path to the image_dataset directory, which includes a train subdirectory, in
    which we have a folder per category we want to classify. The function will generate,
    a val subdirectly, in which we move a portion of the training images
    
    split: fraction of each category that we move to the validation (val) subdirectory
    """
    
    #Create a val subdirectory
    os.mkdir(path + '\\val')
    
    #Loop through all the categories in the train directory
    for i in os.listdir(path + '\\train'):
        #Create the folder in the val subdirectory
        os.mkdir(path + '\\val\\' + i)
        
        #extract and shuffle all the images
        images = os.listdir(path + '\\train\\' + i)
        shuffle(images)
        
        #Move a fraction of the images to the val directory
        for j in range(int(split*len(images))):
            os.rename(path + '\\train\\' + i + '\\' + images[j], path + '\\val\\' + i + '\\' + images[j])
        
        
    
        
    