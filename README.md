# image-classification

This library contains the methods required to build an image recognition API using transfer learning. The module can be used to extract a training set of images from Google Images, train a transfer learning model built on top of InceptionV3, optimize the hyperparameters of the model using [scikit-optimize library](https://scikit-optimize.github.io/), evaluate the accuracy of the model on a test set and classify images online using a simple Flask API.

For any additional information, please contact samuel.mercier@decathlon.com

## Getting started
Make sure you have python 3 and the required libraries (Tensorflow, dill, Pillow, scikit-optimize, selenium and Flask) properly installed. You can then git clone the project to the desired location
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
Once this set is built, **make sure you take a quick look at the set of images**, to cleanup the dataset and remove the images not relevant to the classification problem at hand. 

### Training the classification model
