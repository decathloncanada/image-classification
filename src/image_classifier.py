# -*- coding: utf-8 -*-
"""
Image classifier trained by transfer learning on top of inception model. We generate
an augmented dataset to increase dataset size.

Additional improvements to consider: batch_norm/droupout, consider models
other than inception (Keras has a few of them), scrape imagenet instead of google
images (using imagenetscraper) and ensemble a number of models

@author: sam
"""

import dill
import pickle

from tensorflow.python.keras.applications.inception_v3 import InceptionV3, decode_predictions, preprocess_input
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.python.keras import backend as K

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

TRAIN_DIR = parentdir + '/data/image_dataset/train'
VAL_DIR = parentdir + '/data/image_dataset/val'
TEST_DIR = parentdir + '/data/image_dataset/test'
AUGMENTED_DIR = parentdir + '/data/augmented_dataset'

class image_classifier():
    
    def __init__(self):
        pass
    
    #optimize the hyperparameters of the model        
    def _hyperparameter_optimization(self, num_iterations=50, save_results=True,
                                     display_plot=False):
        """
        num_iterations: number of hyperparameter combinations we try
        """
        
        #import scikit-optimize libraries
        from skopt import gp_minimize
        from skopt.space import Real, Categorical, Integer
        from skopt.plots import plot_convergence
        from skopt.utils import use_named_args
                
        #declare the hyperparameters search space
        dim_epochs = Integer(low=1, high=10, name='epochs')
        dim_hidden_size = Integer(low=3, high=1000, name='hidden_size')   
        dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                                 name='learning_rate')
        dim_dropout = Real(low=0, high=0.9, name='dropout')
        dim_fine_tuning = Categorical(categories=[True, False], name='fine_tuning')
        dim_nb_layers = Integer(low=1, high=3, name='nb_layers')   
        dim_activation = Categorical(categories=['relu', 'tanh'], name='activation')

        dimensions = [dim_epochs,
                      dim_hidden_size,
                      dim_learning_rate,
                      dim_dropout,
                      dim_fine_tuning,
                      dim_nb_layers,
                      dim_activation]
        
        #read default parameters from last optimization
        try:
            with open(parentdir + '/data/trained_model/hyperparameters_search.pickle', 'rb') as f:
                sr = dill.load(f)
            default_parameters = sr.x
            print('parameters of previous optimization loaded!')

        except:
            #fall back default values
            default_parameters = [5, 1024, 1e-4, 0, False, 1, 'relu']
        
        self.number_iterations = 0
    
        #declare the fitness function
        @use_named_args(dimensions=dimensions)
        def fitness(epochs, hidden_size, learning_rate, dropout, 
                    fine_tuning, nb_layers, activation):
            
            self.number_iterations += 1
            
            #print the hyper-parameters            
            print('epochs:', epochs)
            print('hidden_size:', hidden_size)
            print('learning rate:', learning_rate)
            print('dropout:', dropout)
            print('fine_tuning:', fine_tuning)
            print('nb_layers:', nb_layers)
            print('activation:', activation)
            print()
            
            #fit the model
            self.fit(epochs=epochs, hidden_size=hidden_size, learning_rate=learning_rate, dropout=dropout, 
                    fine_tuning=fine_tuning, nb_layers=nb_layers, activation=activation)
            
            #extract fitness
            fitness = self.history['val_accuracy']
                
            print('CALCULATED FITNESS AT ITERATION', self.number_iterations, 'OF:', fitness)
            print()
            
            K.clear_session()
            del self.model
                
            return -1*fitness
                       
        # optimization
        self.search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=num_iterations,
                            x0=default_parameters)
        
        if save_results:
            with open(parentdir + '/data/trained_model/hyperparameters_dimensions.pickle', 'wb') as f:
                dill.dump(dimensions, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            with open(parentdir + '/data/trained_model/hyperparameters_search.pickle', 'wb') as f:
                dill.dump(self.search_result, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print("Hyperparameter search saved!")
            
        if display_plot:
            plot_convergence(self.search_result)   
            
        #build results dictionary
        results_dict = {dimensions[i].name:self.search_result.x[i] for i in range(len(dimensions))}
        print('Optimal hyperameters found of:')
        print(results_dict)
        print()
        print('Optimal fitness value of:', -float(self.search_result.fun))
    
    #we fit the model given the images in the training set
    def fit(self, learning_rate=1e-4, epochs=5, activation='relu',
            dropout=0, hidden_size=1024, nb_layers=1, steps_per_epoch=40,
            val_steps_per_epoch=5, save_augmented=False, 
            batch_size=20, save_model=True, verbose=True,
            fine_tuning=False, NB_IV3_LAYERS_TO_FREEZE=279):
        
        #load the pretrained model, withoug the classification (top) layers
        base_model = InceptionV3(weights='imagenet', include_top=False)
        
        #We expect the classes to be the name of the folders in the training set
        self.categories = os.listdir(TRAIN_DIR)
        
        #Add the classification layers using Keras functional API
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        for _ in range(nb_layers):
            x = Dense(hidden_size, activation=activation)(x) #Hidden layer for classification
            if dropout > 0:
                x = Dropout(rate=dropout)(x)
            
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
                                           rescale=1./255,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           shear_range=0.1,
                                           zoom_range=[0.9, 1.5],
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           fill_mode='nearest'
                                           )
                
        datagen_val = ImageDataGenerator(rescale=1./255)
        
        #Save the augmented images if we want to
        if save_augmented:
            save_to_dir = AUGMENTED_DIR
        else:
            save_to_dir = None
    
        self.generator_train = datagen_train.flow_from_directory(directory=TRAIN_DIR,
                                                            target_size=(299, 299),
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            save_to_dir=save_to_dir)
        
        self.generator_val = datagen_val.flow_from_directory(directory=VAL_DIR,
                                                            target_size=(299, 299),
                                                            batch_size=batch_size,
                                                            shuffle=False)
        
        #Fit the model
        self.history = model.fit_generator(generator=self.generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  verbose=verbose,
                                  validation_data=self.generator_val,
                                  validation_steps=val_steps_per_epoch)
        
        #Fine-tune the model, if we wish so
        if fine_tuning:
            #declare the first layers as trainable
            for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
                layer.trainable = False
            for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
                layer.trainable = True
            model.compile(optimizer=Adam(lr=learning_rate*0.1),   
                          loss=loss,
                          metrics=['accuracy'])
            
            #Fit the model
            self.history = model.fit_generator(generator=self.generator_train,
                                      epochs=epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      verbose=verbose,
                                      validation_data=self.generator_val,
                                      validation_steps=val_steps_per_epoch)
            
        #Evaluate the model, just to be sure
        self.generator_train.reset()
        self.results = model.evaluate_generator(generator=self.generator_train, steps=steps_per_epoch)
        print(self.results[1])
            
        
        #Save the model
        if save_model:
            model.save(parentdir + '/data/trained_models/trained_model.h5')
            print('Model saved!')
            
        self.model = model
            
    #evaluation of the accuracy of classification on the test set
    def predict(self, path):
        generator_test = ImageDataGenerator().flow_from_directory(directory=path,
                                         target_size=(299, 299),
                                         shuffle=False)
        
        #results = model.predict_generator(generator=generator_test)
        self.test_results = self.model.evaluate_generator(generator=generator_test)
        print('The test accuracy is of', self.test_results[1]*100, '%')

        
if __name__ == '__main__':
    classifier = image_classifier()
    classifier.fit(fine_tuning=False, save_model=False, epochs=5, steps_per_epoch=20,
                   val_steps_per_epoch=3, save_augmented=False)
#    classifier.predict(VAL_DIR)
        
        
        
        
        
        
        
        
        