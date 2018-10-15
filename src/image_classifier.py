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
    
    #optimize the hyperparameters of the model        
    def _hyperparameter_optimization(self, num_iterations=50,
                                     max_nrows=None, num_threads=1, calculated_sports=True,
                                     display_plot=False, test_percentage=0.2, k=5,
                                     specific_countries=None, predict_user_ids=None,
                                     save_results=False, max_users=None):
        """
        fitness_function: 0-->built-in precision+recall, 1-->built-in precision,
                          2-->built-in recall, 3-->manual recall function
        """
        
        #The choice of the fitness function affects the construction of the test set
        if fitness_function == 3:
            USE_BUILT_IN_STATS=False
        else:
            USE_BUILT_IN_STATS=True
        
        self.hyperparameter_opt_conditions = {
                'num_threads': num_threads,
                'test_percentage': test_percentage,
                'use_built_in_stats': USE_BUILT_IN_STATS
                }
        
        #import scikit-optimize libraries
        from skopt import gp_minimize
        from skopt.space import Real, Categorical, Integer
        from skopt.plots import plot_convergence
        from skopt.utils import use_named_args
        
        #read the data
        self.countries, self.data = utils.read_data_S3(calculated_sports=calculated_sports, 
                                                       max_nrows=max_nrows,
                                                       specific_countries=specific_countries)
        
        #declare the hyperparameters search space
        dim_num_components = Integer(low=1, high=60, name='num_components')
        dim_epochs = Integer(low=1, high=20, name='epochs')
        dim_loss = Categorical(categories=['warp', 'bpr'], name='loss')      
        dim_learning_rate = Real(low=1e-3, high=1e-1, prior='log-uniform',
                                 name='learning_rate')
        dim_user_alpha = Real(low=0, high=1e-3, name='user_alpha')
        dim_include_user_features = Categorical(categories=[True, False], 
                                                 name='include_user_features')
        dim_add_user_identity = Categorical(categories=[True, False], 
                                            name='add_user_identity')

        dimensions = [dim_num_components,
                      dim_epochs,
                      dim_loss,
                      dim_learning_rate,
                      dim_user_alpha,
                      dim_include_user_features,
                      dim_add_user_identity]
        
        #read default parameters from last optimization
        try:
            with open(parentdir + '/data/trained_model/hyperparameters_search' + ''.join(['_' + i for i in specific_countries]) + '.pickle', 'rb') as f:
                sr = dill.load(f)
            default_parameters = sr.x
            print('parameters of previous optimization loaded!')

        except:
            #fall back default values
            default_parameters = [2, 5, 'warp', 0.05, 0.0, True, False]
        
        self.number_iterations = 0
    
        #declare the fitness function
        @use_named_args(dimensions=dimensions)
        def fitness(num_components, epochs, loss, learning_rate, user_alpha, 
                    include_user_features, add_user_identity):
            
            self.number_iterations += 1
            
            #print the hyper-parameters            
            print('num components:', num_components)
            print('epochs:', epochs)
            print('loss:', loss)
            print('learning rate:', learning_rate)
            print('user alpha:', user_alpha)
            print('include user features:', include_user_features)
            print('add user identity:', add_user_identity)
            print()
            
            #fit the model
            self.fit(num_components=num_components, epochs=epochs, loss=loss, 
                     learning_rate=learning_rate, user_alpha=user_alpha,
                     include_user_features=include_user_features,
                     add_user_identity=add_user_identity,
                     load_data=False, **self.hyperparameter_opt_conditions)
            
            #calculate fitness
            if fitness_function==0: #sum of built-in precision and fitness
                self._precision_at_k(k=k)
                self._recall_at_k(k=k)
                fitness = self.precision + self.recall
            elif fitness_function==1: #built-in precision
                self._precision_at_k(k=k)
                fitness = self.precision
            elif fitness_function==2: #built-in recall
                self._recall_at_k(k=k)
                fitness = self.recall
            elif fitness_function==3: #manual recall function
                self._statistics_at_k(k=k, calculate_precision=False, calculate_coverage=False, max_users=max_users)
                fitness = self.recall
                
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
            with open(parentdir + '/data/trained_model/hyperparameters_dimensions' + ''.join(['_' + i for i in specific_countries]) + '.pickle', 'wb') as f:
                dill.dump(dimensions, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            with open(parentdir + '/data/trained_model/hyperparameters_search' + ''.join(['_' + i for i in specific_countries]) + '.pickle', 'wb') as f:
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
            print('Model saved!')
            
        self.model = model
            
    #evaluation of the accuracy of classification on the test set
    def predict(self):
        #load the model
        
        
        path_to_test = '..\\Test_images'
        generator_test = ImageDataGenerator().flow_from_directory(directory=path_to_test,
                                         target_size=(299, 299),
                                         shuffle=False)
        
        #results = model.predict_generator(generator=generator_test)
        test_results = self.model.evaluate_generator(generator=generator_test)
        print('The test accuracy is of', test_results[1]*100, '%')
        
if __name__ == '__main__':
    classifier = image_classifier()
    classifier.fit(fine_tuning=True, save_model=False)
        
        
        
        
        
        
        
        
        