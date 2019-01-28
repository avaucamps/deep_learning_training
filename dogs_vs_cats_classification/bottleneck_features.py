from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras.models import Model, Sequential
from keras.preprocessing import image
import os
from pathlib import Path
import numpy as np
import pandas as pd


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = BASE_PATH + '/bottleneck_model.h5'
TRAINING_FEATURES = BASE_PATH + '/bottleneck_features_train.npy'
VALIDATION_FEATURES = BASE_PATH + '/bottleneck_features_validation.npy'
CLASSIFICATION_MODEL_WEIGHTS = BASE_PATH + '/bottleneck_classification_weights.h5'
image_size = (299,299)
image_input_size = (299,299,3)
batch_size = 64


def train_bottleneck_model(train_csv, validation_csv, images_path, filename_column, class_column):
    model = get_InceptionV3_model()
    
    if not os.path.isfile(TRAINING_FEATURES):
        save_bottleneck_training_features(
            model=model,
            train_csv=train_csv,
            images_path=images_path,
            filename_column=filename_column,
            class_column=class_column
        )

    if not os.path.isfile(VALIDATION_FEATURES):
        save_bottleneck_validation_features(
            model=model,
            validation_csv=validation_csv,
            images_path=images_path,
            filename_column=filename_column,
            class_column=class_column
        )
    
    train_classification_model(
        train_csv=train_csv,
        validation_csv=validation_csv,
        class_column=class_column,
        training_features=get_training_features(),
        validation_features=get_validation_features()
    )

    classification_model = get_classification_model(
        input_shape=model.output_shape[1:]
    )
    classification_model.load_weights(CLASSIFICATION_MODEL_WEIGHTS)

    model = Model(inputs=model.input, outputs=classification_model(model.output))
    model.save(MODEL_PATH)


def save_bottleneck_training_features(model, train_csv, images_path, filename_column, class_column):
    train_generator = get_training_generator(
        train_csv=train_csv,
        images_path=images_path,
        filename_column=filename_column,
        class_column=class_column
    )
    
    bottleneck_features_train = model.predict_generator(
        train_generator,
        train_generator.n // batch_size
    )

    with open(TRAINING_FEATURES, 'wb') as training_features_file:
        np.save(training_features_file, bottleneck_features_train)


def save_bottleneck_validation_features(model, validation_csv, images_path, filename_column, class_column):
    validation_generator = get_validation_generator(
        validation_csv=validation_csv,
        images_path=images_path,
        filename_column=filename_column,
        class_column=class_column
    )

    bottleneck_features_validation = model.predict_generator(
        validation_generator,
        validation_generator.n // batch_size
    )

    with open(VALIDATION_FEATURES, 'wb') as validation_features_file:
        np.save(validation_features_file, bottleneck_features_validation)


def get_training_features():
    with open(TRAINING_FEATURES, 'rb') as training_features_file:
        return np.load(training_features_file)


def get_validation_features():
    with open(VALIDATION_FEATURES, 'rb') as validation_features_file:
        return np.load(validation_features_file)


def train_classification_model(train_csv, validation_csv, class_column, training_features, validation_features):
    number_cats_training, number_dogs_training = get_number_of_item_per_class(train_csv, class_column)
    training_labels = np.array([0] * number_cats_training + [1] * number_dogs_training)
    number_cats_valid, number_dogs_valid = get_number_of_item_per_class(validation_csv, class_column)
    validation_labels = np.array([0] * number_cats_valid + [1] * number_dogs_valid)

    classification_model = get_classification_model(input_shape=training_features.shape[1:])
    classification_model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    classification_model.fit(
        training_features,
        training_labels,
        epochs=3,
        batch_size=batch_size,
        validation_data=(validation_features, validation_labels)
    )

    classification_model.save_weights(CLASSIFICATION_MODEL_WEIGHTS)


def get_training_generator(train_csv, images_path, filename_column, class_column):
    df = pd.read_csv(str(train_csv))
    
    return ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        dataframe=df,    
        directory=str(images_path),
        x_col=filename_column,
        y_col=class_column,
        has_ext=True,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )


def get_validation_generator(validation_csv, images_path, filename_column, class_column):
    df = pd.read_csv(str(validation_csv))

    return ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        dataframe=df,    
        directory=str(images_path),
        x_col=filename_column,
        y_col=class_column,
        has_ext=True,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )


def get_InceptionV3_model():
    model = InceptionV3(
        weights = 'imagenet',
        include_top = False,
        input_shape = image_input_size
    )

    return model


def get_classification_model(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model


def get_number_of_item_per_class(csv_file, class_column):
    df = pd.read_csv(csv_file)

    item_count = df.groupby(class_column).size()
    return item_count[0], item_count[1]