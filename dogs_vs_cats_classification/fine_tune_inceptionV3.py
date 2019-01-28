from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.optimizers import SGD
import pandas as pd
import os


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = BASE_PATH + '/inceptionV3_model.h5'
image_size = (299,299)
image_input_size = (299,299,3)
batch_size = 64


def fine_tune_model(train_csv, validation_csv, images_path, filename_column, class_column):
    """ Trains an inceptionV3 model by firstly training only the classification layers and then 
    trains the last two convolution blocks and the classification layers.

    # Arguments:
        train_csv: the csv file with the training images path and their label.
        validation_csv: the csv file with the validation images path and their label.
        images_path: the path where the images are located.
        filename_column: the name of the column corresponding to the filename in the csv files.
        class_column: the name of the column corresponding to the class in the csv files.
    """
    fine_tune_new_model(
        train_csv=train_csv,
        validation_csv=validation_csv,
        images_path=images_path,
        filename_column=filename_column,
        class_column=class_column
    )

    fine_tune_existing_model(
        train_csv=train_csv,
        validation_csv=validation_csv,
        images_path=images_path,
        filename_column=filename_column,
        class_column=class_column
    )


def fine_tune_new_model(train_csv, validation_csv, images_path, filename_column, class_column):
    training_generator = get_training_generator(
        train_csv,
        images_path,
        filename_column,
        class_column
    )

    validation_generator = get_validation_generator(
        validation_csv,
        images_path,
        filename_column,
        class_column
    )

    model = get_inceptionV3_model()
    model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit_generator(
        training_generator,
        training_generator.n // batch_size,
        epochs = 12,
        validation_data = validation_generator,
        validation_steps = validation_generator.n // batch_size
    )

    model.save(str(MODEL_PATH))


def fine_tune_existing_model(train_csv, validation_csv, images_path, filename_column, class_column, model_path=MODEL_PATH):
    training_generator = get_training_generator(
        train_csv,
        images_path,
        filename_column,
        class_column
    )

    validation_generator = get_validation_generator(
        validation_csv,
        images_path,
        filename_column,
        class_column
    )

    model = load_model(str(model_path))

    for layer in model.layers[:249]: layer.trainable = False
    for layer in model.layers[249:]: layer.trainable = True

    optimizer = SGD(lr = 0.0001, momentum=0.9, nesterov=True)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit_generator(
        training_generator,
        training_generator.n // batch_size,
        epochs = 3,
        validation_data = validation_generator,
        validation_steps = validation_generator.n // batch_size
    )

    model.save(str(model_path))


def get_training_generator(train_csv, images_path, filename_column, class_column):
    df = pd.read_csv(str(train_csv))

    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.35,
        height_shift_range=0.35,
        horizontal_flip=True,
        zoom_range=0.35,
        shear_range=0.25,
        fill_mode='nearest',
        vertical_flip=True
    ).flow_from_dataframe(
        dataframe=df,    
        directory=str(images_path),
        x_col = filename_column,
        y_col = class_column,
        has_ext = True,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )


def get_validation_generator(valid_csv, images_path, filename_column, class_column):
    df = pd.read_csv(str(valid_csv))

    return ImageDataGenerator(rescale = 1./255).flow_from_dataframe(
        dataframe=df,    
        directory=str(images_path),
        x_col = filename_column,
        y_col = class_column,
        has_ext = True,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

def get_inceptionV3_model():
    base_model = InceptionV3(
        weights = 'imagenet',
        input_shape = image_input_size,
        include_top = False
    )

    output = base_model.output
    output = GlobalAveragePooling2D()(output)
    output = Dense(1024, activation='relu')(output)
    output = Dropout(0.5)(output)
    predictions = Dense(1, activation='sigmoid')(output)

    model = Model(
        inputs=base_model.input,
        outputs=predictions
    )

    for layer in model.layers: layer.trainable = False

    return model