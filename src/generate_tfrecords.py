# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:27:40 2019

helper to save dataset as TF records. Strongly inspired by:
https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36

@author: AI team
"""

import matplotlib.image as mpimg
import tensorflow as tf
import math
import numpy as np

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
AUTO = tf.data.experimental.AUTOTUNE 

class TfrecordsGenerator:
    
    def __init__(self):
        self._create_graph()

    # Create graph to create tfrecords
    def _create_graph(self):
        self.sess = tf.Session()
        
    def _is_png_image(self, filename):
        ext = os.path.splitext(filename)[1].lower()
        return ext == '.png'

    # Run graph to convert PNG image data to JPEG data
    def _convert_png_to_jpeg(self, img):
         png_enc = tf.image.decode_png(img, channels = 3)
         return tf.image.encode_jpeg(png_enc, format = 'rgb', quality = 100)
        

    def _to_tfrecord(self, img_bytes, label):

        example = tf.train.Example(features = tf.train.Features(feature = {
            'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_bytes])),
            'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
        }))
        return example
    
    #main method, to convert all images
    def convert_image_folder(self, img_folder=parentdir+'/data/image_dataset/train',
                             gcs_ouput=parentdir+'/data/image_dataset/train',
                             shards=16, target_size=(299, 299)):
        # Get all file names of images present in folder
        classes = sorted(os.listdir(img_folder))
        print(classes)
        img_pattern = os.path.join(img_folder, '*/*')
        nb_images = len(tf.gfile.Glob(img_pattern))
        shard_size = math.ceil(1.0 * nb_images / shards)
        print("Pattern matches {} images which will be rewritten as {} .tfrec files containing {} images each.".format(nb_images, shards, shard_size))
        
        def decode_jpeg_and_label(filename):
            bits = tf.read_file(filename)
            image = tf.image.decode_jpeg(bits, channels=3)
            label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep='/')
            label = label.values[-2]
            return image, label
        
        def resize_image(image, label):
            image = tf.image.resize_images(image, size=[*target_size])
            image = tf.reshape(image, [*target_size, 3])
            return image, label
        
        def recompress_image(image, label):
            image = tf.cast(image, tf.uint8)
            image = tf.image.encode_jpeg(image, quality=100, format = 'rgb',
                                         optimize_size=True, chroma_downsampling=False)
            return image, label

        filenames = tf.data.Dataset.list_files(img_pattern) # This also shuffles the images
        dataset = filenames.map(decode_jpeg_and_label, num_parallel_calls=AUTO)
        dataset = dataset.map(resize_image, num_parallel_calls=AUTO)
        dataset = dataset.map(recompress_image, num_parallel_calls=AUTO)
        dataset = dataset.batch(shard_size) # sharding: there will be one "batch" of images per file
        iterator = dataset.make_one_shot_iterator()
        next_shard = iterator.get_next()
        
        print("Writing TFRecords")
        for shard in range(shards):
            image, label = self.sess.run(next_shard)
            # batch size used as shard size here
            shard_size = image.shape[0]
            # good practice to have the number of records in the filename
            filename = os.path.join(gcs_ouput, "{:02d}-{}.tfrec".format(shard, shard_size))
  
            with tf.python_io.TFRecordWriter(filename) as out_file:
                for i in range(shard_size):
                    example = self._to_tfrecord(image[i], # re-compressed image: already a byte string
                                                classes.index(label[i].decode('utf8')))
                    out_file.write(example.SerializeToString())
                print("Wrote file {} containing {} records".format(filename, shard_size))
            
            
if __name__=='__main__':
    transformer = TfrecordsGenerator()
    transformer.convert_image_folder(img_folder=parentdir+'/data/image_dataset/val',
                                     tfrecord_file_name=parentdir+'/data/image_dataset/val.tfrecord')
    