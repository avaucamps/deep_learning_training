from dataset_handler import (prepare_dataset, IMAGES_PATH, TRAIN_CSV, VALID_CSV, 
REDUCED_TRAIN_CSV, REDUCED_VALIDATION_CSV, FILENAME_COLUMN, CLASS_COLUMN)
from fine_tune_inceptionV3 import fine_tune_model, fine_tune_existing_model
from bottleneck_features import train_bottleneck_model, MODEL_PATH as bottleneck_model_path

prepare_dataset()

# print("\n ####### Starting training of inceptionV3 model ####### \n")
# fine_tune_model(
#     train_csv=TRAIN_CSV,
#     validation_csv=VALID_CSV,
#     images_path=IMAGES_PATH,
#     filename_column=FILENAME_COLUMN,
#     class_column=CLASS_COLUMN
# )

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