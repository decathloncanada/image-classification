# -*- coding: utf-8 -*-
"""
Image classifier trained by transfer learning on top of inception model. We generate
an augmented dataset to increase dataset size.

Additional improvements to consider: batch_norm/droupout, consider models
other than inception (Keras has a few of them), scrape imagenet instead of google
images (using imagenetscraper) and ensemble a number of models

@author: sam
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.python.keras.applications.inception_v3 import InceptionV3, decode_predictions, preprocess_input
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.python.keras import backend as K

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

TRAIN_DIR = parentdir + '/Data/image_dataset/train'
TEST_DIR = parentdir + '/Data/image_dataset/test'
AUGMENTED_DIR = '/Data/image_dataset/augmented_dataset'

class image_classifier():
    
    def __init__(self):
        pass
    
    #we fit the model given the images in the training set
    def fit(self, learning_rate=1e-4, epochs=5, steps_per_epoch=20, 
            save_augmented=False, batch_size=20, save_model=True, verbose=True,
            fine_tuning=False, NB_IV3_LAYERS_TO_FREEZE=279):
        
        #Load the pretrained model, withoug the classification (top) layers
        base_model = InceptionV3(weights='imagenet', include_top=False)
        
        #We expect the classes to be the name of the folders in the training set
        self.categories = os.listdir(TRAIN_DIR)
        
        #Add the classification layers using Keras functional API
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x) #Hidden layer for classification
        predictions = Dense(len(self.categories), activation='softmax')(x) #Output layer
        model = Model(inputs=base_model.input, outputs=predictions)
        
        #Set only the top layers as trainable (if we want to do fine-tuning,
        #we can train the base layers as a second step)
        for layer in base_model.layers:
            layer.trainable = False
            
        #Define the optimizer and the loss, and compile the model
        optimizer = Adam(lr=learning_rate)
        loss = 'categorical_crossentropy'       
        model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['categorical_accuracy'])
        
        #Define the dataset augmentation and batch generator
        datagen_train = ImageDataGenerator(rotation_range=180,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           shear_range=0.1,
                                           zoom_range=[0.9, 1.5],
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           fill_mode='nearest'
                                           )
        
        #Save the augmented images if we want to
        if save_augmented:
            save_to_dir = AUGMENTED_DIR
        else:
            save_to_dir = None
    
        generator_train = datagen_train.flow_from_directory(directory=TRAIN_DIR,
                                                            target_size=(299, 299),
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            save_to_dir=save_to_dir)
        
        #Fit the model
        history = model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  verbose=verbose)
        
        #Fine-tune the model, if we wish so
        if fine_tuning:
            #declare the first layers as trainable
            for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
                layer.trainable = False
            for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
                layer.trainable = True
            model.compile(optimizer=Adam(lr=learning_rate/0.1),   
                          loss=loss,
                          metrics=['categorical_accuracy'])
            
            #Fit the model
            history = model.fit_generator(generator=generator_train,
                                      epochs=epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      verbose=verbose)
        
        #Save the model
        if save_model:
            model.save(parentdir + '/data/trained_models/trained_model.h5')
            
        self.model = model
            
    #evaluation of the accuracy of classification on the test set
    def evaluate(self):
        path_to_test = '..\\Test_images'
        generator_test = ImageDataGenerator().flow_from_directory(directory=path_to_test,
                                         target_size=(299, 299),
                                         shuffle=False)
        
        #results = model.predict_generator(generator=generator_test)
        test_results = self.model.evaluate_generator(generator=generator_test)
        print('The test accuracy is of', test_results[1]*100, '%')
        
if __name__ == '__main__':
    classifier = image_classifier()
    classifier.fit(fine_tuning=True)
        
        
        
        
        
        
        
        
        