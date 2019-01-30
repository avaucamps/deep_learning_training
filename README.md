# Dogs vs Cats Classification

For the kaggle competition [dogs vs cats classification](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) 
I implemented two classifiers with Keras.

The first classifier [bottleneck_features_model.py](bottleneck_features_model.py) is an InceptionV3 model trained by firstly extracting 
the bottleneck features of a pre-trained network and then training a new classification network on the extracted features. <br />
The other classifier [inceptionV3_fine_tuned_model.py](inceptionV3_fine_tuned_model.py) is an InceptionV3 model fine-tuned directly on the 
new dataset.

With the model using bottleneck features I achieved a public score of 0.06196 on Kaggle. <br/>
With the inceptionV3 fine-tuned model I achieved a public score of 0.06077 on Kaggle.
