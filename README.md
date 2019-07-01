# Dogs vs Cats Classification

Solution for [Kaggle's Dog vs Cats Classification](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/overview).

## About

> In this competition, you'll write an algorithm to classify whether images contain either a dog or a cat. [source](https://www.kaggle.com/c/dogs-vs-cats)

## Solutions

### 1. Bottleneck Features

Using a pre-trained InceptionV3 model trained by firstly extracting the bottleneck features created from a pass on the dogs-vs-cats dataset and then training a new classification network on the extracted features. [code](bottleneck_features_model.py)

### 2. Fine-tuning pretrained network

Using a pre-trained InceptionV3 model fine-tuned on the dogs-vs-cats dataset. [code](inceptionV3_fine_tuned_model.py)

### Results

> Submissions are scored on the log loss. [source](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/overview/evaluation)

* **Bottleneck features** solution score: 0.06196.
* **Fine-tuning pretrained network** solution score: 0.06077.
