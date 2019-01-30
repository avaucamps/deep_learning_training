from dataset_handler import (prepare_dataset, BASE_PATH, IMAGES_PATH, TRAIN_CSV, VALID_CSV, 
REDUCED_TRAIN_CSV, REDUCED_VALIDATION_CSV, FILENAME_COLUMN, CLASS_COLUMN)
from inceptionV3_fine_tuned_model import fine_tune_model, fine_tune_existing_model, MODEL_PATH as fine_tuned_inceptionV3_model_path
from bottleneck_features_model import train_bottleneck_model, MODEL_PATH as bottleneck_model_path
import os

prepare_dataset()

if not os.path.isfile(fine_tuned_inceptionV3_model_path):
    print("\n ####### Starting training of inceptionV3 model ####### \n")
    fine_tune_model(
        train_csv=TRAIN_CSV,
        validation_csv=VALID_CSV,
        images_path=IMAGES_PATH,
        filename_column=FILENAME_COLUMN,
        class_column=CLASS_COLUMN
    )

if not os.path.isfile(bottleneck_model_path):
    print("\n ####### Starting training of bottleneck inceptionV3 model ####### \n")
    train_bottleneck_model(
        train_csv=REDUCED_TRAIN_CSV,
        validation_csv=REDUCED_VALIDATION_CSV,
        images_path=IMAGES_PATH,
        filename_column=FILENAME_COLUMN,
        class_column=CLASS_COLUMN
    )

    print("\n ####### Starting fine-tuning of bottleneck inceptionV3 model ####### \n")
    fine_tune_existing_model(
        train_csv=TRAIN_CSV,
        validation_csv=VALID_CSV,
        images_path=IMAGES_PATH,
        filename_column=FILENAME_COLUMN,
        class_column=CLASS_COLUMN,
        model_path=bottleneck_model_path
    )