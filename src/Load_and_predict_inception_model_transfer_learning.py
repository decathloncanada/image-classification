# -*- coding: utf-8 -*-
"""
Created on Wed May 16 09:28:31 2018

Let's attempt to do some transfer learning - train the classification layer on
top of the inception v3 models. Future potential improvements on top of simply
retraining the classification layer: fine-tuning, batch_norm/droupout, try models
other than inception (Keras has a few of them), scrape imagenet instead of google
images (using imagenetscraper)

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

SAVE_MODEL = False
BATCH_SIZE = 20
SAVE_AUGMENTED = True
TARGET_SIZE = (299, 299) #Input size of the inception model
NUM_CLASSES = 4
train_dir = '..\\Extract_images'
augmented_dir = '..\\Augmented_data'

#Load the pretrained model, feature extraction part
base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x) #Hidden layer for classification
predictions = Dense(NUM_CLASSES, activation='softmax')(x) #Output layer
model = Model(inputs=base_model.input, outputs=predictions)

#Transfer learning: we only want to train the classification part (in fine-tuning,
#we can also train the first few layers)
for layer in base_model.layers:
    layer.trainable = False

optimizer = Adam(lr=1e-5)
loss = 'categorical_crossentropy'

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['categorical_accuracy'])

#Parameters to create augmented dataset
datagen_train = ImageDataGenerator(
      rotation_range=180,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=[0.9, 1.5],
      horizontal_flip=True,
      vertical_flip=True,
      fill_mode='nearest'
      )

#Save the augmented images if we want to
if SAVE_AUGMENTED:
    save_to_dir = augmented_dir
else:
    save_to_dir = None

#Generator of batches
generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    target_size=TARGET_SIZE,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    save_to_dir=save_to_dir)

#Train the model
epochs = 10
steps_per_epoch = 20
historyhistory = model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch)

#Save the model
if SAVE_MODEL:
    model.save('transfer_learning_model.h5')

#Predict classification for a new image
target_labels = [i for i in next(os.walk(train_dir))[1]]
path_to_test = '..\\Test_images'
generator_test = ImageDataGenerator().flow_from_directory(directory=path_to_test,
                                 target_size=TARGET_SIZE,
                                 shuffle=False)
#results = model.predict_generator(generator=generator_test)
test_results = model.evaluate_generator(generator=generator_test)
print('The test accuracy is of', test_results[1]*100, '%')
#model.predict_generator(generator=generator_test)