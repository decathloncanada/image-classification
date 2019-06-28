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
from PIL import Image
import PIL
import math
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import skopt
import dill
import datetime
import glob
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
from efficientnet import EfficientNetB0, EfficientNetB3
AUTO = tf.data.experimental.AUTOTUNE
# Does the TPU support eager mode?
# No, eager mode uses a new dynamic execution engine, while the TPU uses XLA, which performs static compilation of the execution graph.
# https://cloud.google.com/tpu/docs/faq
# tf.enable_eager_execution()

class Swish(tf.keras.layers.Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'
        
class CheckpointDownloader(object):
    """
    Download current state after each iteration to Google Drive.
    Example usage:
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        from google.colab import auth
        from oauth2client.client import GoogleCredentials
        checkpoint_callback = CheckpointDownloader("./result.pkl")
        skopt.gp_minimize(obj_fun, dims, callback=[checkpoint_callback])
    Parameters
    ----------
    * `checkpoint_path`: location where checkpoint are saved to;
    """
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path

    def __call__(self, res):
        """
        Parameters
        ----------
        * `res` [`OptimizeResult`, scipy object]:
            The optimization as a OptimizeResult object.
        """
        if os.path.exists(self.checkpoint_path):
            print('Uploading checkpoint ' + self.checkpoint_path + ' to Google Drive')
            auth.authenticate_user()
            gauth = GoogleAuth()
            gauth.credentials = GoogleCredentials.get_application_default()
            drive = GoogleDrive(gauth)
            file = drive.CreateFile({'title': 'checkpoint.pkl'})
            file.SetContentFile(self.checkpoint_path)
            file.Upload()

class Image_classifier():

    def __init__(self, tfrecords_folder, batch_size=128, use_TPU=False,
            transfer_model='Inception', load_model=False, legacy=False):
        
        self.current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.parent_dir = os.path.dirname(self.current_dir)
        self.train_dir = os.path.join(self.parent_dir, 'data/image_dataset/train')
        self.val_dir = os.path.join(self.parent_dir, 'data/image_dataset/val')
        self.test_dir = os.path.join(self.parent_dir, 'data/image_dataset/test')
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
        self.use_GPU = tf.test.is_built_with_cuda()
        
        if  not tf.__version__.split('.')[1] == '14' and not legacy:
            raise Exception('This notebook is not compatible with lower version of Tensorflow 1.14, please use legacy mode')
        self.legacy = legacy
        
        # We expect the classes to be the name of the folders in the training set
        self.categories = sorted(os.listdir(self.train_dir))
        print('Classes ({}) :'.format(len(self.categories)))
        print(self.categories)

        train_tfrecords = tf.io.gfile.listdir(
            os.path.join(tfrecords_folder, 'train'))
        self.nb_train_shards = len(train_tfrecords)
        print('Training tfrecords = {}'.format(self.nb_train_shards))
        
        val_tfrecords = tf.io.gfile.listdir(
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
        
        self.nb_test_images = len(tf.io.gfile.glob(os.path.join(self.test_dir, '*/*')))
        print('Test images = {}'.format(self.nb_test_images))
        
        self.training_shard_size = math.ceil(self.nb_train_images/self.nb_train_shards)
        print('Training shard size = {}'.format(self.training_shard_size))

        val_shard_size = math.ceil(nb_val_images/self.nb_val_shards)
        print('Val shard size = {}'.format(val_shard_size))

        print('Training batch size = '+str(self.batch_size))
        self.steps_per_epoch = int(self.nb_train_images / self.batch_size)
        print('Training steps per epochs = '+str(self.steps_per_epoch))

        print('Val batch size = '+str(self.batch_size))
        self.validation_steps = int(nb_val_images / self.batch_size)
        print('Val steps per epochs = '+str(self.validation_steps))

        if transfer_model in ['Inception', 'Xception', 'Inception_Resnet', 'B3']:
            self.target_size = (299, 299)
        else:
            self.target_size = (224, 224)
            
        # As we know SWISH activation function recently published by a team at Google. If you are not familiar with the Swish activation (mathematically, f(x)=x*sigmoid(x)) https://arxiv.org/abs/1710.05941
        def _swish(x):
            return (tf.keras.backend.sigmoid(x) * x)

        tf.keras.utils.get_custom_objects().update({'swish': Swish(_swish)})
        
        if load_model:
            # Useful to avoid clutter from old models / layers.
            tf.keras.backend.clear_session()
            self.model = tf.keras.models.load_model(os.path.join(self.parent_dir, 'data/trained_models/trained_model.h5'))
            print('Model loaded !')

        
    """
    helper functions to load tfrecords. Strongly inspired by
    https://colab.research.google.com/github/GoogleCloudPlatform/training-data-analyst/blob/master/courses/fast-and-lean-data-science/07_Keras_Flowers_TPU_playground.ipynb#scrollTo=LtAVr-4CP1rp
    """
    

    def get_input_dataset(self, is_training, nb_readers):
        
        def _read_tfrecord(example):

            features = {
                'image': tf.io.FixedLenFeature((), tf.string),
                'label': tf.io.FixedLenFeature((), tf.int64),
            }
            example = tf.io.parse_single_example(example, features)
            image = tf.image.decode_jpeg(example['image'], channels=3)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            feature = tf.image.resize(image, [*self.target_size])
            label = tf.cast([example['label']], tf.int32)
            return feature, label

        def _load_dataset(filenames):
            buffer_size = 8 * 1024 * 1024  # 8 MiB per file
            dataset = tf.data.TFRecordDataset(
                filenames, buffer_size=buffer_size)
            return dataset
    
        file_pattern = os.path.join(self.tfrecords_folder, "train/*" if is_training else "val/*")
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)
        # Enable non-determinism only for training.
        options = tf.data.Options()
        options.experimental_deterministic = not is_training
        dataset = dataset.with_options(options)
        dataset = dataset.interleave(_load_dataset, nb_readers, num_parallel_calls=AUTO)
        if is_training:
            # Shuffle only for training.
            dataset = dataset.shuffle(buffer_size=math.ceil(self.training_shard_size*self.nb_train_shards/4))
        dataset = dataset.repeat()
        dataset = dataset.map(_read_tfrecord, num_parallel_calls=AUTO)
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(AUTO)
        return dataset

    def get_batched_dataset(self, is_training, nb_readers):
        
        def _read_tfrecord(example):
            features = {
                'image': tf.FixedLenFeature((), tf.string),
                'label': tf.FixedLenFeature((), tf.int64),
            }
            example = tf.parse_single_example(example, features)
            image = tf.image.decode_jpeg(example['image'], channels=3)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            feature = tf.image.resize_images(image, [*self.target_size])
            label = tf.cast([example['label']], tf.int32)
            return feature, label

        def _load_dataset(filenames):
            buffer_size = 8 * 1024 * 1024  # 8 MiB per file
            dataset = tf.data.TFRecordDataset(
                filenames, buffer_size=buffer_size)
            return dataset
        
        file_pattern = os.path.join(
            self.tfrecords_folder, "train/*" if is_training else "val/*")
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)
        dataset = dataset.apply(tf.data.experimental.parallel_interleave(
                                _load_dataset, cycle_length=nb_readers,
                                sloppy=is_training))
        if is_training:
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(
                buffer_size=math.ceil(self.training_shard_size*self.nb_train_shards/4)))
        else:
            dataset = dataset.repeat()
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
                                _read_tfrecord, batch_size=self.batch_size,
                                num_parallel_calls=AUTO, drop_remainder=True))
        dataset = dataset.prefetch(AUTO)
        return dataset

    def get_training_dataset(self):

        return self.get_batched_dataset(True, self.nb_train_shards) if self.legacy else self.get_input_dataset(True, self.nb_train_shards)
                

    def get_validation_dataset(self):
        
        return self.get_batched_dataset(False, self.nb_val_shards) if self.legacy else self.get_input_dataset(False, self.nb_val_shards)
    
    
    # print examples of images not properly classified...
    # ...inspired by https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/10_Fine-Tuning.ipynb
    def confusion_matrix(self, filename='labelled-heatmap', normalize=False):
        from sklearn.metrics import confusion_matrix
        import plotly.plotly as py
        import plotly.graph_objs as go

        # Predict the classes for the images in the validation set
        cls_pred = self.model.predict(self.get_validation_dataset(), steps=self.validation_steps)
        cls_pred = np.argmax(cls_pred, axis=1)
        tf.keras.backend.clear_session()
        print('Predictions labels loaded')
        
        cls_true = []
        dataset = self.get_validation_dataset()
        get_next = dataset.make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            for _ in range(self.validation_steps):
                _, labels = sess.run(get_next)
                cls_true.extend(labels)
        tf.keras.backend.clear_session()
        print('True labels loaded')

        # Print the confusion matrix.
        cm = confusion_matrix(y_true=cls_true,  # True class for test-set.
                              y_pred=cls_pred)  # Predicted class.
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            filename += '-normalized'
        trace = go.Heatmap(z=cm,
                   x=self.categories,
                   y=self.categories)
        data=[trace]
        py.plot(data, filename=filename) 
        
    # function to plot images...
    # ...inspired by https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/10_Fine-Tuning.ipynb
    def plot_images(self, images, cls_true, cls_pred=None, smooth=True, num_images=9):
        assert len(images) == len(cls_true)
        
        # Create figure with sub-plots.
        if math.sqrt(num_images).is_integer():
            nrows = ncols = int(math.sqrt(num_images))
        else:
            for i in reversed(range(math.ceil(math.sqrt(num_images)))):
                if not num_images % i:
                    nrows = int(num_images/i)
                    ncols = int(i)
                    break
        fig, axes = plt.subplots(nrows, ncols)
    
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
                    np.asarray(cls_true[i]).item()]
    
                # Show true and predicted classes.
                if cls_pred is None:
                    xlabel = "True: {0}".format(cls_true_name)
                else:
                    # Name of the predicted class.
                    cls_pred_name = self.categories[
                        np.asarray(cls_pred[i]).item()]
    
                    xlabel = "True: {0}\nPred: {1}".format(
                        cls_true_name, cls_pred_name)
    
                # Show the classes as the label on the x-axis.
                ax.set_xlabel(xlabel)
    
            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])
    
        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.tight_layout()
        plt.show()
    
    # function to plot error images
    def plot_errors(self):
        
        # Predict the classes for the images in the validation set
        cls_pred = self.model.predict(self.get_validation_dataset(), steps=self.validation_steps)
        cls_pred = np.argmax(cls_pred, axis=1)
        tf.keras.backend.clear_session()
        print('Predictions labels loaded')
        
        cls_true = []
        dataset = self.get_validation_dataset()
        get_next = dataset.make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            for _ in range(self.validation_steps):
                _, labels = sess.run(get_next)
                cls_true.extend(labels)
        tf.keras.backend.clear_session()
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
        tf.keras.backend.clear_session()
        del dataset
        del get_next
        print('Images loaded')
        
        # Plot the 9 images we have loaded and their corresponding classes.
        # We have only loaded 9 images so there is no need to slice those again.
        self.plot_images(images=images,
                    cls_true=[ cls_true[i] for i in random_errors],
                    cls_pred=[ cls_pred[i] for i in random_errors])
        
    def classify_folder(self, path=None):
        if path == None : path = self.test_dir 
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(directory=path, 
                                                       target_size=self.target_size,
                                                       shuffle=False,
                                                       interpolation='bilinear',
                                                       color_mode='rgb',
                                                       class_mode=None,
                                                       batch_size=self.nb_test_images)
        images = []
        for _ in range(len(test_generator)):
            images.append(next(test_generator)) 
        images = np.reshape(np.asarray(images), [-1, *self.target_size, 3])
        print('Test images loaded')
           
        cls_pred =  self.model.predict(images, batch_size=1) # Batch size = 1 to avoid OOM
        tf.keras.backend.clear_session()
        print('Test labels loaded')
        
        for i in range(len(images)):
            top_pred = np.argsort(cls_pred[i])[::-1][:3]
            plt.imshow(images[i], interpolation='spline16')
            # Name of the true class.
            cls_pred_name = np.asarray(self.categories)[top_pred]
            cls_pred_perc = cls_pred[i][top_pred]*100
            xlabel = 'Prediction :\n'
            for (x,y) in zip(cls_pred_name, cls_pred_perc):
                xlabel += '{0}, {1:.2f}%\n'.format(x,y)
            plt.xlabel(xlabel)
            plt.xticks([])
            plt.yticks([])
            plt.show()
 
    def classify_images(self, image_path):
        images = glob.glob( os.path.join(image_path, '*.jpg') )
        for image in images:
            #load the image
            image = Image.open(image)
            #reshape the image
            image = image.resize(self.target_size, PIL.Image.BILINEAR).convert("RGB")
            #convert the image into a numpy array
            image = tf.keras.preprocessing.image.img_to_array(image)
            #rescale the pixels to a 0-1 range
            image = image.astype('float32') / 255.
            # and expend to a size 4 tensor
            image_tensor = np.expand_dims(image, axis=0)
            #make and decode the prediction
            result =  self.model.predict(image_tensor)[0]
            #print image and top predictions
            top_pred = np.argsort(result)[::-1][:3]
            plt.imshow(image, interpolation='spline16')
            # Name of the true class.
            cls_pred_name = np.asarray(self.categories)[top_pred]
            cls_pred_perc = result[top_pred]*100
            xlabel = 'Prediction :\n'
            for (x,y) in zip(cls_pred_name, cls_pred_perc):
                xlabel += '{0}, {1:.2f}%\n'.format(x,y)
            plt.xlabel(xlabel)
            plt.xticks([])
            plt.yticks([])
            plt.show()
        
        
    def evaluate(self):
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(directory=self.test_dir, 
                                                       target_size=self.target_size,
                                                       shuffle=False,
                                                       interpolation='bilinear',
                                                       color_mode='rgb',
                                                       class_mode='sparse',
                                                       batch_size=self.batch_size)

        self.test_results = self.model.evaluate_generator(
                generator=test_generator)
        print('Accuracy of', self.test_results[1]*100, '%')
    
    def save_model(self, path='/data/trained_models'):
        if not os.path.exists(os.path.join(self.parent_dir,path)):
            os.makedirs(os.path.join(self.parent_dir,path))
        self.model.save(os.path.join(os.path.join(self.parent_dir,path), 'trained_model.h5'))
        print('Model saved!')
            
    def extract_SavedModel(self, path='./image_classifier/1/'):
        with tf.keras.backend.get_session() as sess:
                tf.saved_model.simple_save(
                    sess,
                    path,
                    inputs={'input_image': self.model.input},
                    outputs={t.name: t for t in self.model.outputs})
           
    # TODO Warning Keras Tuner is still not finished (Status: pre-alpha.)
# =============================================================================
#     def hyperband(self):
#         
#         from kerastuner.tuners import UltraBand
#         from kerastuner.distributions import Fixed, Boolean, Choice, Range, Logarithmic, Linear
#         
#         epochs = Range(name='epochs', start=1, stop=10)
#         hidden_size = Choice(name='hidden_size', selection=[256, 512, 1024, 2048])
#         learning_rate = Logarithmic(name='learning_rate', start=1e-6, stop=1e-2, num_buckets=10)
#         dropout = Fixed(name='dropout', value=0.9)
#         l2_lambda = Logarithmic(name='learning_rate', start=0, stop=0.1, num_buckets=10)
#         
#         tuner = UltraBand(self.fit(epochs=epochs, 
#                                    hidden_size=hidden_size, 
#                                    learning_rate=learning_rate,
#                                    dropout=dropout, 
#                                    l2_lambda=l2_lambda), 
#                           objective='val_sparse_categorical_accuracy', 
#                           label_names=self.categories)
# 
#         tuner.search(self.get_training_dataset,
#                      validation_data=self.get_validation_dataset)
# =============================================================================
            
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

        # declare the hyperparameters search space
        dim_epochs = skopt.space.Integer(low=1, high=8, name='epochs')
        dim_hidden_size = skopt.space.Integer(low=512, high=2048, name='hidden_size')
        dim_learning_rate = skopt.space.Real(low=1e-6, high=1e-2, prior='log-uniform',
                                 name='learning_rate')
        dim_dropout = skopt.space.Real(low=0, high=0.9, name='dropout')
        dim_l2_lambda = skopt.space.Real(low=1e-6, high=1e-2, prior='log-uniform',
                                 name='l2_lambda')

        dimensions = [dim_epochs,
                      dim_hidden_size,
                      dim_learning_rate,
                      dim_dropout,
                      dim_l2_lambda]

        # read default parameters from last optimization
        try:
            res = skopt.load(os.path.join(self.parent_dir, 'data/trained_models/checkpoint.pkl'))
            x0 = res.x_iters
            y0 = res.func_vals
            start_from_checkpoint = True
            print('Parameters of previous optimization loaded!')
            print(x0)
            print(y0)
        except:
            # fall back default values
            default_parameters = [2, 1024, 5e-4, 0.9, 1e-3]
            start_from_checkpoint = False
            
        if not os.path.exists(self.parent_dir + '/data/trained_models'):
                os.makedirs(self.parent_dir + '/data/trained_models')

        # Set `store_objective`
        # to `False` if your objective function (`.specs['args']['func']`) is
        # unserializable (i.e. if an exception is raised when trying to serialize
        # the optimization result)                
        checkpoint_saver = skopt.callbacks.CheckpointSaver(os.path.join(self.parent_dir, 'data/trained_models/checkpoint.pkl'), store_objective=False)
        checkpoint_dowloader = CheckpointDownloader(os.path.join(self.parent_dir, 'data/trained_models/checkpoint.pkl'))
        verbose = skopt.callbacks.VerboseCallback(n_total=num_iterations)

        # declare the fitness function
        @skopt.utils.use_named_args(dimensions=dimensions)
        def _fitness(epochs, hidden_size, learning_rate, dropout, l2_lambda):
            
            # print the hyper-parameters
            print('Fitnessing hyper-parameters')
            print('epochs:', epochs)
            print('hidden_size:', hidden_size)
            print('learning rate:', learning_rate)
            print('dropout:', dropout)
            print('l2_lambda:', l2_lambda)
            
            # fit the model
            self.fit(epochs=epochs, hidden_size=hidden_size, learning_rate=learning_rate,
                     dropout=dropout, l2_lambda=l2_lambda, min_accuracy=self.min_accuracy)

            # extract fitness
            fitness = self.fitness

            del self.model
            tf.keras.backend.clear_session()
            return -fitness
        
        # optimization
        if start_from_checkpoint:
            print('Continuous fitness')
            search_result = skopt.gp_minimize(func=_fitness,
                                             dimensions=dimensions,
                                             x0=x0,    # already examined values for x
                                             y0=y0,    # observed values for x0
                                             # Expected Improvement.
                                             acq_func='EI',
                                             n_calls=num_iterations,
                                             n_random_starts=n_random_starts,
                                             callback=[checkpoint_saver,checkpoint_dowloader,verbose])
        else:
            print('New fitness')
            search_result = skopt.gp_minimize(func=_fitness,
                                             dimensions=dimensions,
                                             # Expected Improvement.
                                             acq_func='EI',
                                             n_calls=num_iterations,
                                             n_random_starts=n_random_starts,
                                             x0=default_parameters,
                                             callback=[checkpoint_saver,checkpoint_dowloader,verbose])

        if save_results:
            with open(self.parent_dir + '/data/trained_models/hyperparameters_dimensions.pickle', 'wb') as f:
                dill.dump(dimensions, f)

            with open(self.parent_dir + '/data/trained_models/hyperparameters_search.pickle', 'wb') as f:
                dill.dump(search_result.x, f)

            print("Hyperparameter search saved!")

        if display_plot:
            skopt.plots.plot_convergence(search_result)

        # build results dictionary
        results_dict = {dimensions[i].name: search_result.x[i]
                        for i in range(len(dimensions))}
        print('Optimal hyperameters found of:')
        print(results_dict)
        print('Optimal fitness value of:', -float(search_result.fun))       
        
    # we fit the model given the images in the training set
    def fit(self, learning_rate=1e-3, epochs=5, activation='swish', hidden_size=1024, 
            include_class_weight=False, save_model=False, dropout=0.5, verbose=True, 
            fine_tuning=True, l2_lambda=5e-4, min_accuracy=None, logs=None,
            extract_SavedModel=False, bn_after_ac=False):

        # Useful to avoid clutter from old models / layers.
        tf.keras.backend.clear_session()
        
        callbacks = None
        if logs is not None:
            logdir = os.path.join(logs, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            print('Fit log dir : ' + logdir)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)
            callbacks = [tensorboard_callback]

        # if we want stop training when no sufficient improvement in accuracy has been achieved
        if min_accuracy is not None:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='sparse_categorical_accuracy', 
                                                        baseline=min_accuracy)
            if callbacks is None:
                callbacks = [early_stop]
            else:
                callbacks.append(early_stop)

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
            
        def _create_model():
            print('Creating model')
            # load the pretrained model, without the classification (top) layers
            if self.transfer_model == 'Xception':
                base_model = tf.keras.applications.Xception(weights='imagenet',
                                      include_top=False, input_shape=(*self.target_size, 3))
                based_model_last_block = 116  # last block 126, two blocks 116
            elif self.transfer_model == 'Inception_Resnet':
                base_model = tf.keras.applications.InceptionResNetV2(
                        weights='imagenet', include_top=False, input_shape=(*self.target_size, 3))
                based_model_last_block = 287  # last block 630, two blocks 287 
            elif self.transfer_model == 'Resnet':
                base_model = tf.keras.applications.ResNet50(weights='imagenet',
                                      include_top=False, input_shape=(*self.target_size, 3))
                based_model_last_block = 155  # last block 165, two blocks 155
            elif self.transfer_model == 'B0':
                base_model = EfficientNetB0(weights='imagenet',include_top=False,
                                            input_shape=(*self.target_size, 3))
                based_model_last_block = 213  # last block 229, two blocks 213
            elif self.transfer_model == 'B3':
                base_model = EfficientNetB3(weights='imagenet',include_top=False,
                                            input_shape=(*self.target_size, 3))
                based_model_last_block = 354  # last block 370, two blocks 354
            else:
                base_model = tf.keras.applications.InceptionV3(weights='imagenet',
                                      include_top=False, input_shape=(*self.target_size, 3))
                based_model_last_block = 249  # last block 280, two blocks 249
   
            # Set only the top layers as trainable (if we want to do fine-tuning,
            # we can train the base layers as a second step)
            base_model.trainable = False
            
            # Target size infered from the base model
            self.target_size = base_model.input_shape[1:3]
            
            # Add the classification layers using Keras functional API
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            # Hidden layer for classification
            if hidden_size==0:
                x = tf.keras.layers.Dropout(rate=dropout)(x)
            elif bn_after_ac:
                x = tf.keras.layers.Dense(hidden_size, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l=l2_lambda))(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(rate=dropout)(x)
            else:
                x = tf.keras.layers.Dense(hidden_size, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l=l2_lambda))(x)
                # scale: When the next layer is linear (also e.g. nn.relu), this can be disabled since the scaling can be done by the next layer.
                x = tf.keras.layers.BatchNormalization(scale=activation!='relu')(x)
                x = tf.keras.layers.Activation(activation=activation)(x)
                x = tf.keras.layers.Dropout(rate=dropout)(x)
                
            predictions = tf.keras.layers.Dense(len(self.categories),
                                activation='softmax')(x)  # Output layer
            # Define the optimizer and the loss and the optimizer
            loss = 'sparse_categorical_crossentropy'
            metrics = ['sparse_categorical_accuracy']
            optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
            
            return tf.keras.Model(inputs=base_model.input, outputs=predictions), base_model, based_model_last_block, loss, metrics, optimizer
        
        # compile the model and fit the model
        if self.legacy :
            model, base_model, based_model_last_block, loss, metrics, optimizer = _create_model()
            if self.use_TPU:
                resolver = tf.contrib.cluster_resolver.TPUClusterResolver()
                strategy=tf.contrib.tpu.TPUDistributionStrategy(resolver)
                print('Compiling for TPU')
                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)
                print('Fitting')
                history = model.fit(self.get_training_dataset, steps_per_epoch=self.steps_per_epoch, epochs=epochs,
                                    validation_data=self.get_validation_dataset, validation_steps=self.validation_steps,
                                    verbose=verbose, callbacks=callbacks, class_weight=class_weight)
            else:
                print('Compiling for GPU') if self.use_GPU else print('Compiling for CPU')
                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                print('Fitting')
                history = model.fit(self.get_training_dataset(), steps_per_epoch=self.steps_per_epoch, epochs=epochs,
                                    validation_data=self.get_validation_dataset(), validation_steps=self.validation_steps,
                                    verbose=verbose, callbacks=callbacks, class_weight=class_weight)
        else:
            if self.use_TPU:
                tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
                tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
                strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster_resolver, steps_per_run=1)
                
                with strategy.scope():
                    model, base_model, based_model_last_block, loss, metrics, optimizer = _create_model()
                    print('Compiling for TPU')  
                    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                    
            else:
                model, base_model, based_model_last_block, loss, metrics, optimizer =_create_model()
                print('Compiling for GPU') if self.use_GPU else print('Compiling for CPU')
                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                
            print('Fitting')
            history = model.fit(self.get_training_dataset(), steps_per_epoch=self.steps_per_epoch, epochs=epochs,
                                validation_data=self.get_validation_dataset(), validation_steps=self.validation_steps,
                                verbose=verbose, callbacks=callbacks, class_weight=class_weight)
        

        # Fine-tune the model, if we wish so
        if fine_tuning and not model.stop_training:
            print('===========')
            print('Fine-tuning')
            print('===========')
            
            fine_tune_epochs = epochs
            total_epochs =  epochs + fine_tune_epochs
            
            print('Freezing {} layers of {} layers from the base model'.format(
                based_model_last_block, len(base_model.layers)))
            # declare the first layers as trainable
            for layer in model.layers[:based_model_last_block]:
                layer.trainable = False
            for layer in model.layers[based_model_last_block:]:
                layer.trainable = True

            # Fit the model
            # we need to recompile the model for these modifications to take effect with a low learning rate
            if self.legacy:
                print('Recompiling model')
                optimizer = tf.keras.optimizers.Adam(lr=learning_rate*0.1)
                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                if self.use_TPU:
                    print('Fine fitting')
                    history = model.fit(self.get_training_dataset, steps_per_epoch=self.steps_per_epoch, epochs=total_epochs,
                                        validation_data=self.get_validation_dataset, validation_steps=self.validation_steps,
                                        verbose=verbose, callbacks=callbacks, class_weight=class_weight,
                                        initial_epoch=epochs)
                else:
                    print('Fine fitting')
                    history = model.fit(self.get_training_dataset(), steps_per_epoch=self.steps_per_epoch, epochs=total_epochs,
                                        validation_data=self.get_validation_dataset(), validation_steps=self.validation_steps,
                                        verbose=verbose, callbacks=callbacks, class_weight=class_weight,
                                        initial_epoch=epochs)
            else:
                if self.use_TPU:
                    with strategy.scope():
                        print('Recompiling model')
                        optimizer = tf.keras.optimizers.Adam(lr=learning_rate*0.1)
                        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                else:
                    print('Recompiling model')
                    optimizer = tf.keras.optimizers.Adam(lr=learning_rate*0.1)
                    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                
                print('Fine fitting')
                history = model.fit(self.get_training_dataset(), steps_per_epoch=self.steps_per_epoch, epochs=total_epochs,
                                    validation_data=self.get_validation_dataset(), validation_steps=self.validation_steps,
                                    verbose=verbose, callbacks=callbacks, class_weight=class_weight,
                                    initial_epoch=epochs)
            
            

        # Evaluate the model, just to be sure
        self.fitness = history.history['val_sparse_categorical_accuracy'][-1]
        self.model = model
        del history
        del model
        
        # Save the model
        if save_model:
            self.save_model()

        # save model in production format
        if extract_SavedModel:
            self.extract_SavedModel()

    
if __name__ == '__main__':
    classifier = Image_classifier()
#   classifier.fit(save_model=False, epochs=4, hidden_size=222,
#                   learning_rate=0.00024,
#                   fine_tuning=True, transfer_model='Inception_Resnet',
#                   activation='tanh',
#                   include_class_weight=True,
#                   min_accuracy=0.4, extract_SavedModel=True)
#    classifier.confusion_matrix()
#    classifier.plot_errors()
#    classifier._hyperparameter_optimization(num_iterations=20)
