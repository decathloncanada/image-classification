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
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pickle
import random

from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import backend as K
import tensorflow as tf

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
    
    #print examples of images not properly classified...
    #...inspired by https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/10_Fine-Tuning.ipynb
    def confusion_matrix(self):       
        from sklearn.metrics import confusion_matrix     
        
        # Predict the classes for the images in the validation set
        self.generator_val.reset()
        y_pred = self.model.predict_generator(self.generator_val,
                                             steps=self.val_steps_per_epoch)
    
        cls_pred = np.argmax(y_pred,axis=1)
        
        # Print the confusion matrix.
        cm = confusion_matrix(y_true=self.generator_val.classes,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

        print("Confusion matrix:")
        
        # Print the confusion matrix as text.
        print(cm)
        
        # Print the class-names for easy reference.
        for i, class_name in enumerate(list(self.generator_train.class_indices.keys())):
            print("({0}) {1}".format(i, class_name))
    
    #function to plot error images        
    def plot_errors(self):
        #function to plot images...
        #...inspired by https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/10_Fine-Tuning.ipynb
        def plot_images(images, cls_true, cls_pred=None, smooth=True):
            assert len(images) == len(cls_true)
        
            # Create figure with sub-plots.
            fig, axes = plt.subplots(3, 3)
        
            # Adjust vertical spacing.
            if cls_pred is None:
                hspace = 0.3
            else:
                hspace = 0.6
            fig.subplots_adjust(hspace=hspace, wspace=0.3)
        
            # Interpolation type.
            if smooth:
                interpolation = 'spline16'
            else:
                interpolation = 'nearest'
        
            for i, ax in enumerate(axes.flat):
                # There may be less than 9 images, ensure it doesn't crash.
                if i < len(images):
                    # Plot image.
                    ax.imshow(images[i],
                              interpolation=interpolation)
        
                    # Name of the true class.
                    cls_true_name = list(self.generator_train.class_indices.keys())[cls_true[i]]
        
                    # Show true and predicted classes.
                    if cls_pred is None:
                        xlabel = "True: {0}".format(cls_true_name)
                    else:
                        # Name of the predicted class.
                        cls_pred_name = list(self.generator_train.class_indices.keys())[cls_pred[i]]
        
                        xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
        
                    # Show the classes as the label on the x-axis.
                    ax.set_xlabel(xlabel)
                
                # Remove ticks from the plot.
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Ensure the plot is shown correctly with multiple plots
            # in a single Notebook cell.
            plt.show()
    
        # Predict the classes for the images in the validation set
        self.generator_val.reset()
        y_pred = self.model.predict_generator(self.generator_val,
                                             steps=self.val_steps_per_epoch)
    
        cls_pred = np.argmax(y_pred,axis=1)
        
        cls_test = self.generator_val.classes
        
        errors = (cls_pred != cls_test)

        # Get the file-paths for images that were incorrectly classified.
        image_paths_test = [os.path.join(VAL_DIR, filename) for filename in self.generator_val.filenames]
        image_paths = np.array(image_paths_test)[errors]
    
        # Load 9 images randomly picked
        image_paths = random.sample(list(image_paths), 9)
        images = [plt.imread(path) for path in image_paths]
        images = np.asarray(images)
        
        # Get the predicted classes for those images.
        cls_pred = cls_pred[errors]
    
        # Get the true classes for those images.
        cls_true = cls_test[errors]
        
        # Plot the 9 images we have loaded and their corresponding classes.
        # We have only loaded 9 images so there is no need to slice those again.
        plot_images(images=images,
                    cls_true=cls_true[0:9],
                    cls_pred=cls_pred[0:9])
        
    
    #optimize the hyperparameters of the model        
    def _hyperparameter_optimization(self, num_iterations=30, save_results=True,
                                     display_plot=False, batch_size=20, use_TPU=False,
                                     transfer_model='Inception'):
        """
        num_iterations: number of hyperparameter combinations we try
        """
        self.batch_size = batch_size
        self.use_TPU = use_TPU
        self.transfer_model = transfer_model
        
        #import scikit-optimize libraries
        from skopt import gp_minimize
        from skopt.space import Real, Categorical, Integer
        from skopt.plots import plot_convergence
        from skopt.utils import use_named_args
                
        #declare the hyperparameters search space
        dim_epochs = Integer(low=1, high=10, name='epochs')
        dim_hidden_size = Integer(low=6, high=2048, name='hidden_size')   
        dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                                 name='learning_rate')
        dim_dropout = Real(low=0, high=0.9, name='dropout')
        dim_fine_tuning = Categorical(categories=[True, False], name='fine_tuning')
        dim_nb_layers = Integer(low=1, high=3, name='nb_layers')   
        dim_activation = Categorical(categories=['relu', 'tanh'], name='activation')
        dim_include_class_weight = Categorical(categories=[True, False], name='include_class_weight')

        dimensions = [dim_epochs,
                      dim_hidden_size,
                      dim_learning_rate,
                      dim_dropout,
                      dim_fine_tuning,
                      dim_nb_layers,
                      dim_activation,
                      dim_include_class_weight]
        
        #read default parameters from last optimization
        try:
            with open(parentdir + '/data/trained_model/hyperparameters_search.pickle', 'rb') as f:
                sr = dill.load(f)
            default_parameters = sr.x
            print('parameters of previous optimization loaded!')

        except:
            #fall back default values
            default_parameters = [5, 1024, 1e-4, 0, False, 1, 'relu', True]
        
        self.number_iterations = 0
    
        #declare the fitness function
        @use_named_args(dimensions=dimensions)
        def fitness(epochs, hidden_size, learning_rate, dropout, 
                    fine_tuning, nb_layers, activation, include_class_weight):
            
            self.number_iterations += 1
            
            #print the hyper-parameters            
            print('epochs:', epochs)
            print('hidden_size:', hidden_size)
            print('learning rate:', learning_rate)
            print('dropout:', dropout)
            print('fine_tuning:', fine_tuning)
            print('nb_layers:', nb_layers)
            print('activation:', activation)
            print('include_class_weight', include_class_weight)
            print()
            
            #fit the model
            self.fit(epochs=epochs, hidden_size=hidden_size, learning_rate=learning_rate, dropout=dropout, 
                    fine_tuning=fine_tuning, nb_layers=nb_layers, activation=activation,
                    include_class_weight=include_class_weight, batch_size=self.batch_size,
                    use_TPU=self.use_TPU, transfer_model=self.transfer_model)
            
            #extract fitness
            fitness = self.fitness
                
            print('CALCULATED FITNESS AT ITERATION', self.number_iterations, 'OF:', fitness)
            print()
            
            del self.model
            K.clear_session()
            
            return -1*fitness
                       
        # optimization
        self.search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=num_iterations,
                            x0=default_parameters)
        
        if save_results:
            if not os.path.exists(parentdir + '/data/trained_models'):
                os.makedirs(parentdir + '/data/trained_models')
                
            with open(parentdir + '/data/trained_models/hyperparameters_dimensions.pickle', 'wb') as f:
                dill.dump(dimensions, f)
            
            with open(parentdir + '/data/trained_models/hyperparameters_search.pickle', 'wb') as f:
                dill.dump(self.search_result.x, f)
            
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
            dropout=0, hidden_size=1024, nb_layers=1, include_class_weight=False,
            save_augmented=False, batch_size=20, save_model=False, verbose=True,
            fine_tuning=False, NB_IV3_LAYERS_TO_FREEZE=279, use_TPU=False,
            transfer_model='Inception'):
        
        #load the pretrained model, without the classification (top) layers
        if transfer_model=='Xception':
            base_model = Xception(weights='imagenet', include_top=False, input_shape=(299,299,3))
            target_size=(299,299)
        elif transfer_model=='Inception_Resnet':
            base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299,299,3))
            target_size=(299,299)
        elif transfer_model=='Resnet':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
            target_size=(224,224)
        else:
            base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
            target_size=(299,299)
        
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
        loss = 'categorical_crossentropy'  
        if use_TPU:
            #if we want to try out the TPU, it looks like we currently need to use
            #tensorflow optimizers...see https://stackoverflow.com/questions/52940552/valueerror-operation-utpu-140462710602256-varisinitializedop-has-been-marked
            #...and https://www.youtube.com/watch?v=jgNwywYcH4w
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)    
            tpu_optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
            model.compile(optimizer=tpu_optimizer,
                  loss=loss,
                  metrics=['categorical_accuracy'])
            
            TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
            model = tf.contrib.tpu.keras_to_tpu_model(model,
                                                      strategy=tf.contrib.tpu.TPUDistributionStrategy(
                                                              tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
            tf.logging.set_verbosity(tf.logging.INFO)
            
        else:
            optimizer = Adam(lr=learning_rate)
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
                                                            target_size=target_size,
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            save_to_dir=save_to_dir)
        
        self.generator_val = datagen_val.flow_from_directory(directory=VAL_DIR,
                                                            target_size=target_size,
                                                            batch_size=batch_size,
                                                            shuffle=False)
        
        steps_per_epoch = self.generator_train.n / batch_size
        self.val_steps_per_epoch = self.generator_val.n / batch_size
        
        #if we want to weight the classes given the imbalanced number of images
        if include_class_weight:
            from sklearn.utils.class_weight import compute_class_weight
            cls_train = self.generator_train.classes
            class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(cls_train),
                                    y=cls_train)
        else:
            class_weight = None
        
        #Fit the model
        history = model.fit_generator(generator=self.generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  verbose=verbose,
                                  class_weight=class_weight,
                                  validation_data=self.generator_val,
                                  validation_steps=self.val_steps_per_epoch)
        
        #Fine-tune the model, if we wish so
        if fine_tuning:
            #declare the first layers as trainable
            for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
                layer.trainable = False
            for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
                layer.trainable = True
                
            model.compile(optimizer=Adam(lr=learning_rate*0.1),   
                          loss=loss,
                          metrics=['categorical_accuracy'])
            
            #Fit the model
            history = model.fit_generator(generator=self.generator_train,
                                      epochs=epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      verbose=verbose,
                                      class_weight=class_weight,
                                      validation_data=self.generator_val,
                                      validation_steps=self.val_steps_per_epoch)
            
        #Evaluate the model, just to be sure
        self.fitness = history.history['val_categorical_accuracy'][-1]
             
        #Save the model
        if save_model:
            if not os.path.exists(parentdir + '/data/trained_models'):
                os.makedirs(parentdir + '/data/trained_models')
            model.save(parentdir + '/data/trained_models/trained_model.h5')
            print('Model saved!')
        
        self.model = model
        del history
        del model
            
    #evaluation of the accuracy of classification on the test set
    def evaluate(self, path, transfer_model='Inception'):
        if transfer_model in ['Inception', 'Xception', 'Inception_Resnet']:
            target_size = (299, 299)
        else:
            target_size = (224, 224)
            
        generator_test = ImageDataGenerator().flow_from_directory(directory=path,
                                         target_size=target_size,
                                         shuffle=False)
        
        #results = model.predict_generator(generator=generator_test)
        self.test_results = self.model.evaluate_generator(generator=generator_test)
        print('Accuracy of', self.test_results[1]*100, '%')

        
if __name__ == '__main__':
    classifier = image_classifier()
    classifier.fit(save_model=False, save_augmented=False, transfer_model='Inception', epochs=1)
    classifier.confusion_matrix()
    classifier.plot_errors()        
#    classifier._hyperparameter_optimization(num_iterations=20)