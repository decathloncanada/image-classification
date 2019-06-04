# -*- coding: utf-8 -*-

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
import os, math, inspect
import numpy as np
import tensorflow as tf
AUTO = tf.data.experimental.AUTOTUNE
tf.logging.set_verbosity(tf.logging.INFO)

class Tfrecords_image_classifier:
    
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.parent_dir = os.path.dirname(self.current_dir)
        self.train_dir = os.path.join(self.parent_dir, 'data/image_dataset/train')
        self.val_dir = os.path.join(self.parent_dir, 'data/image_dataset/val')
        # We expect the classes to be the name of the folders in the training set
        self.categories = sorted(os.listdir(self.train_dir))
    
    def fit(self, tfrecords_folder, learning_rate=1e-3,
            epochs=5, activation='relu', dropout=0, hidden_size=1024, nb_layers=1,
            include_class_weight=False, batch_size=20, save_model=False, verbose=True,
            fine_tuning=False, layers_to_freeze_ratio=0.5, use_TPU=False,
            transfer_model='Inception_Resnet', min_accuracy=None, extract_SavedModel=False,
            nb_cpu_cores=8, epsilon=1e-08, callbacks=None):
        
        # Useful to avoid clutter from old models / layers.
        K.clear_session()

        if use_TPU and batch_size % 8:
            print('Batch size {} is not multiple of 8, required for TPU'.format(batch_size))
            batch_size = 8 * round(batch_size/8)
            print('New batch size is {}'.format(batch_size))

        train_tfrecords = tf.gfile.ListDirectory(
            os.path.join(tfrecords_folder, 'train'))
        nb_train_shards = len(train_tfrecords)
        print('Training tfrecords = {}'.format(nb_train_shards))
        
        val_tfrecords = tf.gfile.ListDirectory(
            os.path.join(tfrecords_folder, 'val'))
        nb_val_shards = len(val_tfrecords)
        print('Val tfrecords = {}'.format(nb_val_shards))
        
        nb_train_images = 0
        for train_tfrecord in train_tfrecords:
            nb_train_images += int(train_tfrecord.split('.')[0].split('-')[1])
        print('Training images = '+str(nb_train_images))
        
        nb_val_images = 0
        for val_tfrecord in val_tfrecords:
            nb_val_images += int(val_tfrecord.split('.')[0].split('-')[1])
        print('Val images = '+str(nb_val_images))

        training_shard_size = math.ceil(nb_train_images/nb_train_shards)
        print('Training shard size = {}'.format(training_shard_size))

        val_shard_size = math.ceil(nb_val_images/nb_val_shards)
        print('Val shard size = {}'.format(val_shard_size))

        print('Training batch size = '+str(batch_size))
        steps_per_epoch = int(nb_train_images / batch_size)
        print('Training steps per epochs = '+str(steps_per_epoch))

        print('Val batch size = '+str(batch_size))
        validation_steps = int(nb_val_images / batch_size)
        print('Val steps per epochs = '+str(validation_steps))

        if transfer_model in ['Inception', 'Xception', 'Inception_Resnet']:
            target_size = (299, 299)
        else:
            target_size = (224, 224)
            
        def read_tfrecord(example):

            features = {
                'image': tf.FixedLenFeature((), tf.string),
                'label': tf.FixedLenFeature((), tf.int64),
            }
            example = tf.parse_single_example(example, features)
            image = tf.image.decode_jpeg(example['image'], channels=3)
            if use_TPU:
                image = tf.image.convert_image_dtype(image, dtype=tf.bfloat16)
            else:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            feature = tf.reshape(image, [*target_size, 3])
            label = tf.cast([example['label']], tf.int32)
            return feature, label

        def load_dataset(filenames):
            buffer_size = 8 * 1024 * 1024  # 8 MiB per file
            dataset = tf.data.TFRecordDataset(
                filenames, buffer_size=buffer_size)
            return dataset

        def get_batched_dataset(tfrecords_folder, is_training, nb_readers):
            file_pattern = os.path.join(
                tfrecords_folder, "train/*" if is_training else "val/*")
            dataset = tf.data.Dataset.list_files(
                file_pattern, shuffle=is_training)
            dataset = dataset.apply(tf.data.experimental.parallel_interleave(
                                    load_dataset, cycle_length=nb_readers,
                                    sloppy=is_training))
            if is_training:
                dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(
                    buffer_size=nb_train_images))
            else:
                dataset = dataset.repeat()
            dataset = dataset.apply(tf.data.experimental.map_and_batch(
                                    read_tfrecord, batch_size=batch_size,
                                    num_parallel_calls=AUTO, drop_remainder=True))
            dataset = dataset.prefetch(AUTO)
            return dataset

        def get_training_dataset():
            return get_batched_dataset(tfrecords_folder, True, nb_train_shards)

        def get_validation_dataset():
            return get_batched_dataset(tfrecords_folder, False, nb_val_shards)
        
        # if we want stop training when no sufficient improvement in accuracy has been achieved
        if min_accuracy is not None:
            callback = EarlyStopping(
                monitor='sparse_categorical_accuracy', baseline=min_accuracy)
            if callbacks is None:
                callbacks = [callback]
            else:
                callbacks.append(callback)

        # load the pretrained model, without the classification (top) layers
        if transfer_model == 'Xception':
            base_model = Xception(weights='imagenet',
                                  include_top=False, input_shape=(299, 299, 3))
        elif transfer_model == 'Inception_Resnet':
            base_model = InceptionResNetV2(
                weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        elif transfer_model == 'Resnet':
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
        for _ in range(nb_layers):
            x = Dense(hidden_size, activation=activation)(
                x)  # Hidden layer for classification
            if dropout > 0:
                x = Dropout(rate=dropout)(x)

        predictions = Dense(len(self.categories),
                            activation='softmax')(x)  # Output layer

        if use_TPU:
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
            print('Total labels ({}) :'.format(len(cls_train)))
            print('Unique labels ({}) :'.format(len(np.unique(cls_train))))
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
        if use_TPU:
            tpu = tf.contrib.cluster_resolver.TPUClusterResolver()  # TPU detection
            strategy = tf.contrib.tpu.TPUDistributionStrategy(tpu)
            model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)
            # Little wrinkle: reading directly from dataset object not yet implemented
            # for Keras/TPU. Please use a function that returns a dataset.
            history = model.fit(get_training_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                validation_data=get_validation_dataset, validation_steps=validation_steps,
                                verbose=verbose, callbacks=callbacks, class_weight=class_weight)
        else:
            history = model.fit(get_training_dataset(), steps_per_epoch=steps_per_epoch, epochs=epochs,
                                validation_data=get_validation_dataset(), validation_steps=validation_steps,
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
            if use_TPU:
                print('TPU fine fit')
                # Little wrinkle: reading directly from dataset object not yet implemented
                # for Keras/TPU. Please use a function that returns a dataset.
                history = model.fit(get_training_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                    validation_data=get_validation_dataset, validation_steps=validation_steps,
                                    verbose=verbose, callbacks=callbacks, class_weight=class_weight)
            else:
                print('CPU/GPU fine fit')
                history = model.fit(get_training_dataset(), steps_per_epoch=steps_per_epoch, epochs=epochs,
                                    validation_data=get_validation_dataset(), validation_steps=validation_steps,
                                    verbose=verbose, callbacks=callbacks, class_weight=class_weight)
                
        # Evaluate the model, just to be sure
        self.fitness = history.history['val_sparse_categorical_accuracy'][-1]
        
        # Save the model
        if save_model:
            if not os.path.exists(self.parent_dir + '/data/trained_models'):
                os.makedirs(self.parent_dir + '/data/trained_models')
            model.save(self.parent_dir + '/data/trained_models/trained_model.h5')
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
    