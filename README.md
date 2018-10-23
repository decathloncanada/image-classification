# image-classification

This library contains the methods required to build an image recognition API using transfer learning. The module can be used to extract a training set of images from Google image, train a transfer learning model built on top of InceptionV3, optimize the hyperparameters of the model using [scikit-optimize library](https://scikit-optimize.github.io/), evaluate the accuracy of the model on a training set and classify images online using a simple Flash API.

For any additional information, please contact samuel.mercier@decathlon.com

## Getting started
1. git clone the project to the desired location
```
git clone git@github.com:decathloncanada/image-classification.git
```

2. Make sure you have python 3 and the required libraries (Tensorflow, dill, Pillow, scikit-optimize, selenium and Flask) properly installed
