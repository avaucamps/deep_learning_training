from pathlib import Path
import os
import numpy as np
import pandas as pd
import collections


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGES_PATH = Path(BASE_PATH + '/data/train')
DATA_PATH = Path(BASE_PATH + '/data')
TRAIN_CSV = DATA_PATH/'training_dataset.csv'
VALID_CSV = DATA_PATH/'validation_dataset.csv'
REDUCED_TRAIN_CSV = DATA_PATH/'reduced_training_dataset.csv'
REDUCED_VALIDATION_CSV = DATA_PATH/'reduced_validation_dataset.csv'
REDUCED_TRAINING_DATASET_SIZE = 2560
REDUCED_VALIDATION_DATASET_SIZE = 768
FILENAME_COLUMN = 'filename'
CLASS_COLUMN = 'class'
TEST_PATH = Path(BASE_PATH + '/test/test')
TEST_CSV = Path(BASE_PATH + '/test/test.csv')


def prepare_dataset():
    """ Creates training and validation CSV files with the full dataset.
    Also creates reduced training and validation CSV files with 10% of the dataset.
    """

    if os.path.isfile(str(TRAIN_CSV)) and \
       os.path.isfile(str(VALID_CSV)) and \
       os.path.isfile(str(REDUCED_TRAIN_CSV)) and \
       os.path.isfile(str(REDUCED_VALIDATION_CSV)) and \
       os.path.isfile(str(TEST_CSV)):
        return

    training_dict, validation_dict = get_training_and_validation_dictionaries(IMAGES_PATH)

    training_keys = list(training_dict.keys())
    training_values = list(training_dict.values())
    training_images_per_class = int(REDUCED_TRAINING_DATASET_SIZE/2)
    reduced_training_keys = training_keys[:training_images_per_class] + training_keys[-training_images_per_class:]
    reduced_training_values = training_values[:training_images_per_class] + training_values[-training_images_per_class:]

    create_csv(TRAIN_CSV, training_keys, training_values)
    create_csv(REDUCED_TRAIN_CSV, reduced_training_keys, reduced_training_values)

    validation_keys = list(validation_dict.keys())
    validation_values = list(validation_dict.values())
    validation_images_per_class = int(REDUCED_VALIDATION_DATASET_SIZE/2)
    reduced_validation_keys = validation_keys[:validation_images_per_class] + validation_keys[-validation_images_per_class:]
    reduced_validation_values = validation_values[:validation_images_per_class] + validation_values[-validation_images_per_class:]

    create_csv(VALID_CSV, validation_keys, validation_values)
    create_csv(REDUCED_VALIDATION_CSV, reduced_validation_keys, reduced_validation_values)
    
    test_filenames = get_test_filenames(TEST_PATH)
    create_csv(TEST_CSV, test_filenames)


def create_csv(path, keys, values=None):
    if values:
        df = pd.DataFrame(
            {
                FILENAME_COLUMN: keys,
                CLASS_COLUMN: values
            },
            columns = [FILENAME_COLUMN, CLASS_COLUMN]
        )
    else:
        df = pd.DataFrame(
            {
                FILENAME_COLUMN: keys
            },
            columns=[FILENAME_COLUMN]
        )

    df.to_csv(str(path), index=False)


def get_training_and_validation_dictionaries(dataset_path):
    cross_validation_indexes = get_cross_validation_indexes(get_dataset_size(dataset_path))

    index = 0
    training_dict = collections.OrderedDict()
    validation_dict = collections.OrderedDict()
    for i in dataset_path.iterdir():
        filename = str(i).split('\\')[-1]

        if index in cross_validation_indexes:
            validation_dict[filename] = get_class(filename)
        else:
            training_dict[filename] = get_class(filename)

        index += 1

    return training_dict, validation_dict


def get_test_filenames(test_dataset_path):
    filenames = []
    for i in test_dataset_path.iterdir():
        split_filepath = str(i).split('\\')
        filename = split_filepath[-1]
        filenames.append(filename.split('.')[0])

    filenames.sort(key=int)
    for i in range(len(filenames)):
        filenames[i] = filenames[i] + '.jpg'

    return filenames


def get_dataset_size(dataset_path):
    dataset_path = str(dataset_path)
    size = 0

    for name in os.listdir(dataset_path):
        if os.path.isfile(dataset_path + '/' + name):
            size += 1
    
    return size


def get_cross_validation_indexes(dataset_size, validation_set_percentage = 0.2):
    number_of_values = int(dataset_size * validation_set_percentage)
    indexes = np.random.permutation(dataset_size)

    return indexes[0:number_of_values]


def get_class(filename):
    strings = str(filename).split('.')
    return strings[0]