# image-classification

This library contains the methods required to build an image recognition API using transfer learning. The module can be used to extract a training set of images from Google Images, train a transfer learning model built on top of InceptionV3, optimize the hyperparameters of the model using [scikit-optimize library](https://scikit-optimize.github.io/), evaluate the accuracy of the model on a test set and classify images online using a simple Flask API. The library capitalizes on the concepts of data augmentation, fine tuning and hyperparameter optimization, to achieve high accuracy given small sets of training images. 

For any additional information, please contact samuel.mercier@decathlon.com

## Getting started
Make sure you have python 3 and the required libraries (Tensorflow, dill, Pillow, scikit-optimize, selenium and Flask) properly installed. If you want to use the extract_images functionality, make sure you have your [chromedriver executable](http://chromedriver.chromium.org/) in the root folder. 

You can then git clone the project to the desired location
```
git clone git@github.com:decathloncanada/image-classification.git
```

## Services
This library performs the following tasks: extraction of a set of images from Google Images, training a neural network built by transfer learning, optimization of the neural network hyperparameters, evaluation of the model accuracy on a test set, and classification of images using a Flask API.

### Building a training set of images
To build a set of images to train your neural network, first edit the "searchterms.csv" document in the data folder to indicate the categories you want to classify, the search terms from which you would like to build your sets of images, along with the number of images you want to extract from the Google search. For instance, let's assume you want to build a training and a validation set to distinguish a hockey player from a soccer player. As an example, the searchterms.csv file could look as follows:

| search_term | number_imgs | folder_name | train_val |
| ------------- |-------------|-------------|-------------|
| hockey player  | 150 | hockey_player | train |
| soccer player  | 150 | soccer_player | train |
| joueur de hockey | 20 | hockey_player | val |
| joueur de soccer | 20 | soccer_player | val |

In this example, we capitalize on the fact that searching equivalent terms in different languages (in this case English, *hockey player*, and French, *joueur de hockey*) to extract different images for the training set and the validation set.

Then, you can run the following command:
```
python main.py --task extract_images
```
This will extract the desired number of images for each search term, and store them in a data/image_dataset directory organized as follows:
```
data/
  image_dataset/
    train/
      hockey_player/
        hockey_player_1.jpg
        hockey_player_2.jpg
        ...
      soccer_player/
        soccer_player_1.jpg
        soccer_player_2.jpg
        ...
    val/
      hockey_player/
        hockey_player_1.jpg
        hockey_player_2.jpg
        ...
      soccer_player/
        soccer_player_1.jpg
        soccer_player_2.jpg
        ...
        
```
Once this set is built, **make sure you take a quick look at the images**, to cleanup the dataset and remove the images not relevant to the classification problem at hand. Note that you can have multiple search terms for a given class of image, and that there is no limit to the number of different categories you want your model to classify. Note again that if you want to use this functionality, make sure to have your [chromedriver executable](http://chromedriver.chromium.org/) in the root directory.

### Training the classification model
Once you have a dataset of images (in a train and a val directory, structured as described in the previous section) for each of your classes, you can train your custom made classifier by running the following command:
```
python main.py --task fit --save_model 1
```
This will train the neural network using the images in the dataset, and provide the training and validation accuracy. The hyperparameters (number of epochs, number of hidden layers, size of the hidden layers, learning rate, dropout rate, fine tuning, activation function and weighting images given the number of images in each class) used are those stored after hyperparameter optimization (see the following section), or default values if such a file is not found. The trained model will be saved in a the '/data/trained_models/trained_model.h5' file, unless you provide a *--save_model 0* argument.  

### Optimization of the hyperparameters
The classification system contains a number of hyperparameters (number of epochs, number of hidden layers, size of the hidden layers, learning rate, dropout rate, fine tuning, activation function and weighting images given the number of images in each class) whose values strongly affect the accuracy of the classifier. Values of these hyperparameters appropriate for to the categories we want to classify can be found by the following command:
```
python main.py --task hyperparameters --number_iterations 20
```
This calls an hyperparameter optimization function which, using [scikit-optimize](https://scikit-optimize.github.io/), tries different combinations of the hyperparameters to find values maximizing the accuracy of the classifier. The argument --number_iterations indicates the number of different combinations of the hyperparameters we let scikit-optimize tries to find good values of the hyperparameters. 

The optimal hyperparameters are saved as a hyperparameters_search.pickle file in the ./data/trained_model directory. Optimization of the hyperparameters can take a few hours (depending on the number of classes and the number of images per class), as many calls to the model .fit function are required to identify quality hyperparameter values.

### Evaluation of the model accuracy
The classification accuracy of the model can be assessed on a given set of images by the following command:
```
python main.py --task evaluate --evaluate_directory test
```
This will load the model saved in the ./data/trained_model directory, classify the images found in the directory (either train, val or test) specified by the --evaluate_directory argument, and return the classification accuracy.

### Classifying an image: from main.py
The class to which an image belongs can be predicted by running the following command:
```
python main.py --task classify --img {IMAGE_PATH}
```
where you replace {IMAGE_PATH} with the path of the image that you want to classify. The call will return the probability that the image belongs to each class, as classified by the model saved in the ./data/trained_model directory. For instance, let's say you want to distinguish hockey players from soccer players. You can extract a training set of images following the *Building a training set of images* section, and train a model (using the default values of the hyperparameters) following the *Training the classification model* section. Calling the classify method on the following image:

![alt text](https://rdsmedia.cookieless.ca/sports/hockey/nhl/player/212x296/xt.fss.l.nhl.com-p.5497.jpg)

should return category: probability value pairs such as:
```
{'hockey_player': 0.9165782, 'soccer_player': 0.08342175}
```
The classifier successfully identified the greater probability that the image describes a hockey player. Note that the response could be slightly different from the one above, given the randomness in neural network initialization.

### Classifying an image: API call
A simple API was built using Flask framework to classify images online. This API can be run locally by the following command:
```
python app.py
```
The API is then available at http://localhost:5000/classify. The API takes one argument (image), describing the path to the image we want to classify. A POST request can be run as follows

```
curl -X POST -F img=@{IMAGE_PATH} 'http://localhost:5000/classify'
```
where {IMAGE_PATH} is the path of the image that you want to classify. For instance, calling the API to classify the image presented in the previous section returns the following json:

```
{
  "predictions":[
    {
    "label":"hockey_player",
    "probability":0.9165781736373901
    },
    {
    "label":"soccer_player",
    "probability":0.0834217518568039
    }
  ],
  "success":true
}

```
## Roadmap
Future works will involve the application of this library to develop a scalable sport recognition API and sport product detection API. Follow https://developers.decathlon.com for updates.
