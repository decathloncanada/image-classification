# -*- coding: utf-8 -*-
"""
Image classifier trained by transfer learning on top of inception model. We generate
an augmented dataset to increase dataset size.

Additional improvements to consider: batch_norm/droupout, consider models
other than inception (Keras has a few of them), scrape imagenet instead of google
images (using imagenetscraper) and ensemble a number of models

This file: an attempt to read tf records

@author: AI team
"""
import inspect
import sys
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
import math
import random
import pickle as pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import dill
import tensorflow as tf
AUTO = tf.data.experimental.AUTOTUNE
tf.logging.set_verbosity(tf.logging.INFO)
# Does the TPU support eager mode?
# No, eager mode uses a new dynamic execution engine, while the TPU uses XLA, which performs static compilation of the execution graph.
# https://cloud.google.com/tpu/docs/faq
# tf.enable_eager_execution()


class image_classifier():

    def __init__(self, tfrecords_folder, batch_size=128, use_TPU=False,
            transfer_model='Inception', load_model=False):
        
        self.current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.parent_dir = os.path.dirname(self.current_dir)
        self.train_dir = os.path.join(self.parent_dir, 'data/image_dataset/train')
        self.val_dir = os.path.join(self.parent_dir, 'data/image_dataset/val')
        # We expect the classes to be the name of the folders in the training set
        self.categories = sorted(os.listdir(self.train_dir))
        self.tfrecords_folder = tfrecords_folder
        
        if use_TPU and batch_size % 8:
            print('Batch size {} is not multiple of 8, required for TPU'.format(batch_size))
            batch_size = 8 * round(batch_size/8)
            print('New batch size is {}'.format(batch_size))
        
        self.batch_size = batch_size
        self.use_TPU = use_TPU
        self.transfer_model = transfer_model
        
        # We expect the classes to be the name of the folders in the training set
        self.categories = sorted(os.listdir(self.train_dir))
        print('Classes ({}) :'.format(len(self.categories)))
        print(self.categories)

        train_tfrecords = tf.gfile.ListDirectory(
            os.path.join(tfrecords_folder, 'train'))
        self.nb_train_shards = len(train_tfrecords)
        print('Training tfrecords = {}'.format(self.nb_train_shards))
        
        val_tfrecords = tf.gfile.ListDirectory(
            os.path.join(tfrecords_folder, 'val'))
        self.nb_val_shards = len(val_tfrecords)
        print('Val tfrecords = {}'.format(self.nb_val_shards))
        
        self.nb_train_images = 0
        for train_tfrecord in train_tfrecords:
            self.nb_train_images += int(train_tfrecord.split('.')[0].split('-')[1])
        print('Training images = '+str(self.nb_train_images))
        
        nb_val_images = 0
        for val_tfrecord in val_tfrecords:
            nb_val_images += int(val_tfrecord.split('.')[0].split('-')[1])
        print('Val images = '+str(nb_val_images))

        training_shard_size = math.ceil(self.nb_train_images/self.nb_train_shards)
        print('Training shard size = {}'.format(training_shard_size))

        val_shard_size = math.ceil(nb_val_images/self.nb_val_shards)
        print('Val shard size = {}'.format(val_shard_size))

        print('Training batch size = '+str(self.batch_size))
        self.steps_per_epoch = int(self.nb_train_images / self.batch_size)
        print('Training steps per epochs = '+str(self.steps_per_epoch))

        print('Val batch size = '+str(self.batch_size))
        self.validation_steps = int(nb_val_images / self.batch_size)
        print('Val steps per epochs = '+str(self.validation_steps))

        if transfer_model in ['Inception', 'Xception', 'Inception_Resnet']:
            self.target_size = (299, 299)
        else:
            self.target_size = (224, 224)
        
        if load_model:
            try:
                # Useful to avoid clutter from old models / layers.
                K.clear_session()
                self.model = tf.keras.models.load_model(os.path.join(self.parent_dir, 'data/trained_models/trained_model.h5'))
                print('Model loaded !')
                # self.model.summary()
            except:
                print('Loading model error')

        
    """
    helper functions to load tfrecords. Strongly inspired by
    https://colab.research.google.com/github/GoogleCloudPlatform/training-data-analyst/blob/master/courses/fast-and-lean-data-science/07_Keras_Flowers_TPU_playground.ipynb#scrollTo=LtAVr-4CP1rp
    """
    def read_tfrecord(self,example):

        features = {
            'image': tf.FixedLenFeature((), tf.string),
            'label': tf.FixedLenFeature((), tf.int64),
        }
        example = tf.parse_single_example(example, features)
        image = tf.image.decode_jpeg(example['image'], channels=3)
        if self.use_TPU:
            image = tf.image.convert_image_dtype(image, dtype=tf.bfloat16)
        else:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        feature = tf.reshape(image, [*self.target_size, 3])
        label = tf.cast(example['label'], tf.int32)
        return feature, label

    def load_dataset(self,filenames):
        buffer_size = 8 * 1024 * 1024  # 8 MiB per file
        dataset = tf.data.TFRecordDataset(
            filenames, buffer_size=buffer_size)
        return dataset

    def get_batched_dataset(self, is_training, nb_readers):
        file_pattern = os.path.join(
            self.tfrecords_folder, "train/*" if is_training else "val/*")
        dataset = tf.data.Dataset.list_files(
            file_pattern, shuffle=is_training)
        dataset = dataset.apply(tf.data.experimental.parallel_interleave(
                                self.load_dataset, cycle_length=nb_readers,
                                sloppy=is_training))
        if is_training:
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(
                buffer_size=self.nb_train_images))
        else:
            dataset = dataset.repeat()
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
                                self.read_tfrecord, batch_size=self.batch_size,
                                num_parallel_calls=AUTO, drop_remainder=True))
        dataset = dataset.prefetch(AUTO)
        return dataset

    def get_training_dataset(self):
        return self.get_batched_dataset(True, self.nb_train_shards)

    def get_validation_dataset(self):
        return self.get_batched_dataset(False, self.nb_val_shards)
    
    
    # print examples of images not properly classified...
    # ...inspired by https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/10_Fine-Tuning.ipynb
    def confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        import plotly.plotly as py
        import plotly.graph_objs as go

        # Predict the classes for the images in the validation set
        cls_pred = self.model.predict(self.get_validation_dataset(), steps=self.validation_steps)
        cls_pred = np.argmax(cls_pred, axis=1)
        K.clear_session()
        print('Predictions labels loaded')
        
        cls_true = []
        dataset = self.get_validation_dataset()
        get_next = dataset.make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            for _ in range(self.validation_steps):
                _, labels = sess.run(get_next)
                cls_true.extend(labels)
        K.clear_session()
        print('True labels loaded')

        # Print the confusion matrix.
        cm = confusion_matrix(y_true=cls_true,  # True class for test-set.
                              y_pred=cls_pred)  # Predicted class.
        trace = go.Heatmap(z=cm,
                   x=self.categories,
                   y=self.categories)
        data=[trace]
        py.iplot(data, filename='labelled-heatmap') 
        
    # function to plot error images
    def plot_errors(self):
        
        # function to plot images...
        # ...inspired by https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/10_Fine-Tuning.ipynb
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
                    cls_true_name = self.categories[
                        cls_true[i]]

                    # Show true and predicted classes.
                    if cls_pred is None:
                        xlabel = "True: {0}".format(cls_true_name)
                    else:
                        # Name of the predicted class.
                        cls_pred_name = self.categories[
                            cls_pred[i]]

                        xlabel = "True: {0}\nPred: {1}".format(
                            cls_true_name, cls_pred_name)

                    # Show the classes as the label on the x-axis.
                    ax.set_xlabel(xlabel)

                # Remove ticks from the plot.
                ax.set_xticks([])
                ax.set_yticks([])

            # Ensure the plot is shown correctly with multiple plots
            # in a single Notebook cell.
            plt.show()

        # Predict the classes for the images in the validation set
        cls_pred = self.model.predict(self.get_validation_dataset(), steps=self.validation_steps)
        cls_pred = np.argmax(cls_pred, axis=1)
        K.clear_session()
        print('Predictions labels loaded')
        
        cls_true = []
        dataset = self.get_validation_dataset()
        get_next = dataset.make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            for _ in range(self.validation_steps):
                _, labels = sess.run(get_next)
                cls_true.extend(labels)
        K.clear_session()
        del dataset
        del get_next
        print('True labels loaded')
        
        # get all errors index
        errors = []
        for i in range(len(cls_pred)):
            if cls_pred[i]!=cls_true[i]:
                errors.append(i)

        # Load 9 images randomly picked
        random_errors = sorted(random.sample(errors, 9))
        print('random_errors :')
        print(random_errors)
        
        images = []
        dataset = self.get_validation_dataset()
        get_next = dataset.make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            for i in range(self.validation_steps):
                features, _ = sess.run(get_next)
                for j in range(self.batch_size):
                    if self.batch_size*i+j in random_errors:
                        images.append(features[j])  
        K.clear_session()
        del dataset
        del get_next
        print('Images loaded')
        
        # Plot the 9 images we have loaded and their corresponding classes.
        # We have only loaded 9 images so there is no need to slice those again.
        plot_images(images=images,
                    cls_true=[ cls_true[i] for i in random_errors],
                    cls_pred=[ cls_pred[i] for i in random_errors])

    # optimize the hyperparameters of the model

    def hyperparameter_optimization(self, num_iterations=20, save_results=True,
                                     display_plot=False, n_random_starts=10,
                                     cutoff_regularization=False, min_accuracy=None):
        """
        min_accuracy: minimum value of categorical accuracy we want after 1 iteration
        num_iterations: number of hyperparameter combinations we try (aim for a 1:1 to 2:1 ration num_iterations/n_random_starts)   
        n_random_starts: number of random combinations of hyperparameters first tried
        """

        self.min_accuracy = min_accuracy

        # import scikit-optimize libraries
        from skopt import gp_minimize
        from skopt.space import Real, Categorical, Integer
        from skopt.plots import plot_convergence
        from skopt.utils import use_named_args

        # declare the hyperparameters search space
        dim_epochs = Integer(low=1, high=10, name='epochs')
        dim_hidden_size = Integer(low=256, high=2048, name='hidden_size')
        dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                                 name='learning_rate')
        dim_nb_layers = Integer(low=1, high=3, name='nb_layers')
        dim_activation = Categorical(
            categories=['relu', 'tanh'], name='activation')

        # TODO maybe hyperparameters to add :
        # epsilon
        # freeze ratio

        dimensions = [dim_epochs,
                      dim_hidden_size,
                      dim_learning_rate,
                      dim_nb_layers,
                      dim_activation]

        # read default parameters from last optimization
        try:
            with open(self.parentdir + '/data/trained_model/hyperparameters_search.pickle', 'rb') as f:
                sr = dill.load(f)
            default_parameters = sr.x
            print('parameters of previous optimization loaded!')

        except:
            # fall back default values
            default_parameters = [5, 1024, 1e-3, 1, 'tanh']

        self.number_iterations = 0

        # declare the fitness function
        @use_named_args(dimensions=dimensions)
        def fitness(self, epochs, hidden_size, learning_rate, nb_layers, activation):

            self.number_iterations += 1

            # print the hyper-parameters
            print('epochs:', epochs)
            print('hidden_size:', hidden_size)
            print('learning rate:', learning_rate)
            print('nb_layers:', nb_layers)
            print('activation:', activation)
            # fit the model
            self.fit(epochs=epochs, hidden_size=hidden_size, learning_rate=learning_rate,
                     nb_layers=nb_layers, activation=activation, min_accuracy=self.min_accuracy)

            # extract fitness
            fitness = self.fitness

            print('CALCULATED FITNESS AT ITERATION',
                  self.number_iterations, 'OF:', fitness)
            print()

            del self.model
            K.clear_session()

            return -1*fitness

        # optimization
        self.search_result = gp_minimize(func=fitness,
                                         dimensions=dimensions,
                                         # Expected Improvement.
                                         acq_func='EI',
                                         n_calls=num_iterations,
                                         n_random_starts=n_random_starts,
                                         x0=default_parameters)

        if save_results:
            if not os.path.exists(self.parentdir + '/data/trained_models'):
                os.makedirs(self.parentdir + '/data/trained_models')

            with open(self.parentdir + '/data/trained_models/hyperparameters_dimensions.pickle', 'wb') as f:
                dill.dump(dimensions, f)

            with open(self.parentdir + '/data/trained_models/hyperparameters_search.pickle', 'wb') as f:
                dill.dump(self.search_result.x, f)

            print("Hyperparameter search saved!")

        if display_plot:
            plot_convergence(self.search_result)

        # build results dictionary
        results_dict = {dimensions[i].name: self.search_result.x[i]
                        for i in range(len(dimensions))}
        print('Optimal hyperameters found of:')
        print(results_dict)
        print()
        print('Optimal fitness value of:', -float(self.search_result.fun))

    # we fit the model given the images in the training set
    def fit(self, learning_rate=1e-3, epochs=5, activation='tanh', hidden_size=1024, 
            nb_layers=1, include_class_weight=True, save_model=False, dropout=0,
            verbose=True, fine_tuning=True, layers_to_freeze_ratio=0.5, 
            min_accuracy=None, extract_SavedModel=False, epsilon=1e-08, callbacks=None):

        # Useful to avoid clutter from old models / layers.
        K.clear_session()

        # if we want stop training when no sufficient improvement in accuracy has been achieved
        if min_accuracy is not None:
            callback = EarlyStopping(
                monitor='sparse_categorical_accuracy', baseline=min_accuracy)
            if callbacks is None:
                callbacks = [callback]
            else:
                callbacks.append(callback)

        # load the pretrained model, without the classification (top) layers
        if self.transfer_model == 'Xception':
            base_model = Xception(weights='imagenet',
                                  include_top=False, input_shape=(299, 299, 3))
        elif self.transfer_model == 'Inception_Resnet':
            base_model = InceptionResNetV2(
                weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        elif self.transfer_model == 'Resnet':
            base_model = ResNet50(weights='imagenet',
                                  include_top=False, input_shape=(224, 224, 3))
        else:
            base_model = InceptionV3(
                weights='imagenet', include_top=False, input_shape=(299, 299, 3))

        nb_layers_base_model = len(base_model.layers)
        print('Base model layers = '+str(nb_layers_base_model))

        # Add the classification layers using Keras functional API
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # Hidden layer for classification
        for _ in range(nb_layers):
            x = Dense(hidden_size, activation=activation)(x)  
            # x = BatchNormalization()(x)

        predictions = Dense(len(self.categories),
                            activation='softmax')(x)  # Output layer

        if self.use_TPU:
            with tf.contrib.tpu.bfloat16_scope():
                model = Model(inputs=base_model.input, outputs=predictions)
        else:
            model = Model(inputs=base_model.input, outputs=predictions)

        # Set only the top layers as trainable (if we want to do fine-tuning,
        # we can train the base layers as a second step)
        for layer in base_model.layers:
            layer.trainable = False

        # if we want to weight the classes given the imbalanced number of images
        if include_class_weight:
            cls_train = []
            from sklearn.utils.class_weight import compute_class_weight
            for dir in os.listdir(self.train_dir):
                for file in os.listdir(os.path.join(self.train_dir, dir)):
                    cls_train.append(dir)
            print('Total labels ({})'.format(len(cls_train)))
            print('Unique labels ({})'.format(len(np.unique(cls_train))))
            class_weight = compute_class_weight(class_weight='balanced',
                                                classes=sorted(
                                                    np.unique(cls_train)),
                                                y=sorted(cls_train))
            print('Classes weight :')
            print(class_weight)
        else:
            class_weight = None

        # Define the optimizer and the loss, and compile the model
        loss = 'sparse_categorical_crossentropy'
        metrics = ['sparse_categorical_accuracy']
        optimizer = Adam(lr=learning_rate, epsilon=epsilon)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Fit the model
        if self.use_TPU:
            tpu = tf.contrib.cluster_resolver.TPUClusterResolver()  # TPU detection
            strategy = tf.contrib.tpu.TPUDistributionStrategy(tpu)
            model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)
            # Little wrinkle: reading directly from dataset object not yet implemented
            # for Keras/TPU. Please use a function that returns a dataset.
            history = model.fit(self.get_training_dataset, steps_per_epoch=self.steps_per_epoch, epochs=epochs,
                                validation_data=self.get_validation_dataset, validation_steps=self.validation_steps,
                                verbose=verbose, callbacks=callbacks, class_weight=class_weight)
        else:
            history = model.fit(self.get_training_dataset(), steps_per_epoch=self.steps_per_epoch, epochs=epochs,
                                validation_data=self.get_validation_dataset(), validation_steps=self.validation_steps,
                                verbose=verbose, callbacks=callbacks, class_weight=class_weight)

        # Fine-tune the model, if we wish so
        if fine_tuning and not model.stop_training:
            print('============')
            print('Begin fine-tuning')
            print('============')

            # Add more epochs to train longer at lower lr
            epochs *= 2

            nb_layers_to_freeze = math.ceil(
                nb_layers_base_model*layers_to_freeze_ratio)
            print('Freezing {} layers of {} layers from the base model'.format(
                nb_layers_to_freeze, nb_layers_base_model))
            # declare the first layers as trainable
            for layer in model.layers[:nb_layers_to_freeze]:
                layer.trainable = False
            for layer in model.layers[nb_layers_to_freeze:]:
                layer.trainable = True

            print('Recompiling model')
            optimizer = Adam(lr=learning_rate*0.1, epsilon=epsilon)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            # Fit the model
            if self.use_TPU:
                print('TPU fine fit')
                # Little wrinkle: reading directly from dataset object not yet implemented
                # for Keras/TPU. Please use a function that returns a dataset.
                history = model.fit(self.get_training_dataset, steps_per_epoch=self.steps_per_epoch, epochs=epochs,
                                    validation_data=self.get_validation_dataset, validation_steps=self.validation_steps,
                                    verbose=verbose, callbacks=callbacks, class_weight=class_weight)
            else:
                print('CPU/GPU fine fit')
                history = model.fit(self.get_training_dataset(), steps_per_epoch=self.steps_per_epoch, epochs=epochs,
                                    validation_data=self.get_validation_dataset(), validation_steps=self.validation_steps,
                                    verbose=verbose, callbacks=callbacks, class_weight=class_weight)

        # Evaluate the model, just to be sure
        self.fitness = history.history['val_sparse_categorical_accuracy'][-1]

        # Save the model
        if save_model:
            if not os.path.exists(self.parentdir + '/data/trained_models'):
                os.makedirs(self.parentdir + '/data/trained_models')
            model.save(self.parentdir + '/data/trained_models/trained_model.h5')
            print('Model saved!')

        # save model in production format
        if extract_SavedModel:
            export_path = "./image_classifier/1/"

            with K.get_session() as sess:
                tf.saved_model.simple_save(
                    sess,
                    export_path,
                    inputs={'input_image': model.input},
                    outputs={t.name: t for t in model.outputs})

        else:
            self.model = model
            del history
            del model

    # TODO evaluation of the accuracy of classification on the test set
    def evaluate(self, path, transfer_model='Inception'):
        if transfer_model in ['Inception', 'Xception', 'Inception_Resnet']:
            target_size = (299, 299)
        else:
            target_size = (224, 224)

        generator_test = ImageDataGenerator().flow_from_directory(directory=path,
                                                                  target_size=target_size,
                                                                  shuffle=False)

        #results = model.predict_generator(generator=generator_test)
        self.test_results = self.model.evaluate_generator(
            generator=generator_test)
        print('Accuracy of', self.test_results[1]*100, '%')


if __name__ == '__main__':
    classifier = image_classifier()
    classifier.fit(save_model=False, epochs=4, hidden_size=222,
                   learning_rate=0.00024,
                   fine_tuning=True, transfer_model='Inception_Resnet',
                   activation='tanh',
                   include_class_weight=True,
                   min_accuracy=0.4, extract_SavedModel=True)
#    classifier.confusion_matrix()
#    classifier.plot_errors()
#    classifier._hyperparameter_optimization(num_iterations=20)
