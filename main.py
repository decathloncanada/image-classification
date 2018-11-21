# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:36:37 2018

Main function to train an algorithm and-or classify images from a trained model

@author: AI team
"""
import argparse
import dill
import io
import numpy as np
import operator
import os
from PIL import Image
import PIL

from src import extract_images as ext
from src import image_classifier as ic

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras import models

#extract the arguments
parser = argparse.ArgumentParser(description='Extarct images, run hyperperameter search, fit the classifier, evalute the accuracy or predict the class')
parser.add_argument('--task', type=str, default='pass',
                    help="""
                    task to perform: 
                    extract_images-->save images from a Google images search
                    hyperparameters --> optimize classifier hyperparameters; 
                    fit-->fit the classifier (and optionaly save) the classifier; 
                    evaluate-->­­­calculate the accuracy on a given set of images
                    classify-->predict the probability that the image is from the possible classes
                    """)
parser.add_argument('--save_model', type=int, default=0,
                    help="""
                    If we want (1) or not (0) to save the model we are fitting
                    """)
parser.add_argument('--number_iterations', type=int, default=20,
                    help="""
                    Number of iteractions to perform when doing an hyperparameter optimization. As to be greater than 10
                    """)
parser.add_argument('--img', type=str, default=None,
                    help="""
                    Path of the image when we want to predict its class
                    """)
parser.add_argument('--evaluate_directory', type=str, default='test',
                    help="""
                    If we want to evaluate accuracy based on images in the "train", "val" or "test" directory
                    """)
parser.add_argument('--batch_size', type=int, default=20,
                    help="""
                    Batch size of the classifier
                    """)
parser.add_argument('--transfer_model', type=str, default='Inception',
                    help="""
                    Base model used for classification - Inception (V3) and Xception currently supported
                    """)
parser.add_argument('--use_TPU', type=int, default=0,
                    help="""
                    If we want (1) or not (0) to fit the model using a TPU
                    """)

args = parser.parse_args()

#verify the format of the arguments
if args.task not in ['extract_images', 'hyperparameters', 'fit', 'evaluate', 'classify', None]:
    print('Task not supported')
    args.task = 'pass'

if args.task == 'evalute_directory':    
    if args.evaluate_directory not in ['train', 'val', 'test']:
        print('evaluate_directory has to be train, val or test')
        args.task = 'pass'

if args.task == 'fit':
    if args.save_model == 1:
        save_model=True
    elif args.save_model == 0:
        save_model=False
    else:
        print('save_model argument is not 0 or 1')
        args.task = 'pass'
    
if args.use_TPU == 1:
    use_TPU=True
elif args.use_TPU == 0:
    use_TPU=False
else:
    print('use_TPU argument is not 0 or 1')
    args.task = 'pass'

if not (args.number_iterations > 10 and isinstance(args.number_iterations, int)):
    print('number_iterations has to be an integer greater than 10')
    args.task = 'pass'
    
if not (args.batch_size > 0 and isinstance(args.batch_size, int)):
    print('batch_size has to be a positive integer')
    args.task = 'pass'

if args.task == 'classify':    
    if os.path.exists(args.img):
        img_path = args.img
    else:
        print('Unknown path')
        args.task = 'pass'
        
if args.transfer_model not in ['Inception', 'Xception']:    
    print(args.transfer_model + ' not supported. transfer_model supported: Inception and Xception')
    args.task = 'pass'

#function to preprocess the image
def prepare_image(image):
    #reshape the image
    image = image.resize((299,299), PIL.Image.BILINEAR).convert("RGB")
    #convert the image into a numpy array, and expend to a size 4 tensor
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    #rescale the pixels to a 0-1 range
    image = image.astype(np.float32)/255
    return image

#function to translate the predictions into probability values
def decode_prediction(results):
    #get the name of classes
    classes = list(os.walk('data/image_dataset/train'))[0][1]
    #build and order the results dictionary
    result_dict = {classes[i]: results[i] for i in range(len(classes))}
    result_dict = dict(sorted(result_dict.items(), key=operator.itemgetter(1), reverse=True))
    return result_dict

#function to classify the images in a given directory
def classify(img_path):
    #load the image
    image = Image.open(img_path)
    #preprocess the image
    image = prepare_image(image)
    #load the model
    model = models.load_model('data/trained_models/trained_model.h5')
    #make and decode the prediction
    results =  model.predict(image)[0]
    print(decode_prediction(results))
    
#function to perform data extraction
def extract_images():
    extracter = ext.extract_images()
    extracter.run(verbose=True)
    
#function to perform hyperparameter optimization
def hyperparameters():
    classifier = ic.image_classifier()
    classifier._hyperparameter_optimization(num_iterations=args.number_iterations,
                                            batch_size=args.batch_size,
                                            use_TPU=use_TPU,
                                            transfer_model=args.transfer_model)
    
#function to fit the model using saved hyperparameters (when available) 
def fit():
    try:
        #read the optimized hyperparameters
        with open('./data/trained_models/hyperparameters_dimensions.pickle', 'rb') as f:
                dimensions = dill.load(f)
                
        with open('./data/trained_models/hyperparameters_search.pickle', 'rb') as f:
                sr = dill.load(f)
                
        opt_params = {dimensions[i].name:sr[i] for i in range(len(dimensions))}
        
    except:
        print('Could not find optimal hyperparameters. Selecting default values')
        opt_params = {}
    
    classifier = ic.image_classifier()
    classifier.fit(save_model=save_model, use_TPU=use_TPU, 
                   transfer_model=args.transfer_model,
                   **opt_params)
    
#function to evaluate the classification accuracy
def evaluate():
    #load the model
    model = models.load_model('data/trained_models/trained_model.h5')
    #build the generator
    generator = ImageDataGenerator(rescale=1./255).flow_from_directory(directory='data/image_dataset/' + args.evaluate_directory,
                                         target_size=(299, 299),
                                         shuffle=False)
    results = model.evaluate_generator(generator=generator)
    print('Accuracy of', results[1]*100, '%')

#run the proper function given the --task argument passed to the function
if args.task == 'hyperparameters':
    hyperparameters()
    
elif args.task == 'fit':
    fit()
    
elif args.task == 'extract_images':
    extract_images()
    
elif args.task == 'classify':
    classify(img_path)
    
elif args.task == 'evaluate':
    evaluate()