# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:27:40 2019

helper to save dataset as TF records. Strongly inspired by:
https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36

@author: AI team
"""

import matplotlib.image as mpimg
import tensorflow as tf

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

class generate_tfrecords:
    
    def __init__(self):
        self._create_graph()

    # Create graph to convert PNG image data to JPEG data
    def _create_graph(self):
        tf.reset_default_graph()
        self.png_img_pl = tf.placeholder(tf.string)
        png_enc = tf.image.decode_png(self.png_img_pl, channels = 3)
        # Set how much quality of image you would like to retain while conversion
        self.png_to_jpeg = tf.image.encode_jpeg(png_enc, format = 'rgb', quality = 100)

    def _is_png_image(self, filename):
        ext = os.path.splitext(filename)[1].lower()
        return ext == '.png'

    # Run graph to convert PNG image data to JPEG data
    def _convert_png_to_jpeg(self, img):
        sess = tf.get_default_session()
        return sess.run(self.png_to_jpeg, feed_dict = {self.png_img_pl: img})

    def _convert_image(self, img_path, label, classes):
        img_shape = mpimg.imread(img_path).shape
        filename = os.path.basename(img_path).split('.')[0]

        # Read image data in terms of bytes
        with tf.gfile.FastGFile(img_path, 'rb') as fid:
            image_data = fid.read()

            # Encode PNG data to JPEG data
            if self._is_png_image(img_path):
                image_data = self._convert_png_to_jpeg(image_data)
                
#        one_hot_label = tf.tile(tf.expand_dims(label, axis=-1), [len(classes)])
#        one_hot_label = tf.cast(tf.math.equal(one_hot_label, list(range(len(classes)))), tf.uint8)

        example = tf.train.Example(features = tf.train.Features(feature = {
            'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [filename.encode('utf-8')])),
            'rows': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[0]])),
            'cols': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[1]])),
            'channels': tf.train.Feature(int64_list = tf.train.Int64List(value = [3])),
            'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_data])),
            'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label])),
#            'one_hot_label': tf.train.Feature(bytes_list = tf.train.BytesList(value = [one_hot_label.numpy().tobytes()]))
        }))
        return example
    
    #main method, to convert all images
    def convert_image_folder(self, img_folder=parentdir+'/data/image_dataset/train',
                             tfrecord_file_name=parentdir+'/data/image_dataset/train.tfrecord'):
        # Get all file names of images present in folder
        classes = os.listdir(img_folder)
        classes_paths = [os.path.abspath(os.path.join(img_folder, i)) for i in classes]

        with tf.python_io.TFRecordWriter(tfrecord_file_name) as writer:
            for i, j in enumerate(classes):
                #for all the classes, get the list of pictures
                img_paths = os.listdir(classes_paths[i])    
                img_paths = [os.path.abspath(os.path.join(classes_paths[i], x)) for x in img_paths]
                
                for img_path in img_paths:
                    #for all the images, get the tf record
                    example = self._convert_image(img_path, i, classes)
                    writer.write(example.SerializeToString())
                    
if __name__=='__main__':
    transformer = generate_tfrecords()
    transformer.convert_image_folder(img_folder=parentdir+'/data/image_dataset/val',
                                     tfrecord_file_name=parentdir+'/data/image_dataset/val.tfrecord')
    