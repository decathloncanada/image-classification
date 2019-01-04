# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:28:41 2018

Utils functions for the image-classifier

split_train: function to build a validation set from a training set of images.

@author: AI team
"""
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

from random import shuffle

def split_train(path=parentdir+'/data/image_dataset', split=0.1):
    """
    path: path to the image_dataset directory, which includes a train subdirectory, in
    which we have a folder per category we want to classify. The function will generate,
    a val subdirectly, in which we move a portion of the training images
    
    split: fraction of each category that we move to the validation (val) subdirectory
    """
    
    #Create a val subdirectory
    os.mkdir(path + '/val')
    
    #Loop through all the categories in the train directory
    for i in os.listdir(path + '/train'):
        #Create the folder in the val subdirectory
        os.mkdir(path + '/val/' + i)
        
        #extract and shuffle all the images
        images = os.listdir(path + '/train/' + i)
        shuffle(images)
        
        #Move a fraction of the images to the val directory
        for j in range(int(split*len(images))):
            os.rename(path + '/train/' + i + '/' + images[j], path + '/val/' + i + '/' + images[j])

#function to add cutoff regularization to image classification...
#...taken from https://github.com/yu4u/cutout-random-erasing/blob/master/random_eraser.py            
def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=1, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser
        
        
    
        
    