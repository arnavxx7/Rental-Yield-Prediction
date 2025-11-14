import os

PIPELINE_NAME: str = "Rental Yield Prediction"
ARTIFACTS_FOLDER_NAME: str = "Artifacts"
RAW_DATA_FILE_NAME = "rental_data.csv"

TRAIN_FILE_NAME: str = 'Train.csv'
TEST_FILE_NAME: str ='Test.csv'
NEW_DATA_FILE_NAME: str = "New_data.csv"
TRAIN_TEST_SPLIT_RATIO: float = 0.2
DRIFT_REPORT_FILE_NAME: str = "drift_report.yaml"
TRANSFORMED_TRAIN_FILE_NAME: str = "Train.npy"
TRANSFORMED_TEST_FILE_NAME: str = "Test.npy"
TARGET_ENCODING_FILE_NAME: str = "target_encoding.pkl"
ORDINAL_ENCODING_FILE_NAME: str = "ordinal_encoding.pkl"
LABEL_ENCODING_FILE_NAME: str = "label_encoding.pkl"
OHE_FILE_NAME: str = "ohe.pkl"
SCALER_FILE_NAME: str = "scaler.pkl"
X_FILE_NAME: str = "X.csv"
Y_FILE_NAME: str = "y.csv"

DATA_INGESTION_FOLDER_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_FOLDER_NAME: str = "feature_store"
DATA_INGESTION_INGESTED_FOLDER_NAME: str = "ingested"


DATA_VALIDATION_FOLDER_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_FOLDER_NAME: str = "drift_report"
DATA_VALIDATION_VALIDATED_FOLDER_NAME: str = "validated"

SCHEMA_FILE_PATH: str = os.path.join("data_schema", "schema.yaml")


DATA_TRANSFORMATION_FOLDER_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_FOLDER_NAME: str = "transformed_data"
DATA_TRANSFORMATION_TRANSFORMATION_OBJECTS_FOLDERNAME: str = "transformation_objects"
DATA_TRANSFORMATION_IMPUTER_OBJ_FOLDER_NAME: str = "imputer_obj"
DATA_TRANSFORMATION_ENCODING_OBJ_FOLDER_NAME: str = "encoding_obj"
DATA_TRANSFORMATION_SCALER_OBJ_FOLDER_NAME: str = "scaler_obj"

HIGH_CORR_COLS = ['Suburb',
 'Floor',
 'Total_floor',
 'Parking',
 'num_BHK',
 'Furnishing',
 'Veg/Non-veg',
 'num_amenities',
 'age',
 'balconies',
 'area',
 'Family',
 'Family/Bachelor',
 "Rent"]

FEATURES_TO_SCALE = [
    "Floor",
    "Total_floor",
    "Parking",
    "num_BHK",
    "num_amenities",
    "age",
    "balconies",
    "area"
]

FEATURES_TO_LOG_TRANSFORM = [
    "area",
    "Suburb",
    "Rent"
]

MODEL_TRAINER_FOLDER_NAME: str = 'model_training'
MODEL_TRAINER_BEST_MODEL_FOLDER_NAME: str = "best_model"
BEST_MODEL_FILE_NAME: str = "best_model.pkl"
