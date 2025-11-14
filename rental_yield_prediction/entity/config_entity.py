
from rental_yield_prediction.constants import training_pipeline, push_data_constants, prediction_pipeline

from datetime import datetime
import os

class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp = timestamp.strftime("%d_%m_%Y_%H_%M_%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_folder_name = training_pipeline.ARTIFACTS_FOLDER_NAME
        self.artifact_dir = os.path.join(self.artifact_folder_name, timestamp)
        self.timestamp: str = timestamp



class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_FOLDER_NAME)
        self.raw_data_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_FOLDER_NAME, training_pipeline.RAW_DATA_FILE_NAME
            )
        self.train_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_FOLDER_NAME, training_pipeline.TRAIN_FILE_NAME
        )
        self.test_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_FOLDER_NAME, training_pipeline.TEST_FILE_NAME
        )
        self.new_data_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_FOLDER_NAME, training_pipeline.NEW_DATA_FILE_NAME
        )
        self.train_test_split_ratio: float = training_pipeline.TRAIN_TEST_SPLIT_RATIO
        self.postgresql_hostname: str = push_data_constants.HOSTNAME
        self.postgresql_database: str = push_data_constants.DATABASE
        self.postgresql_username: str = push_data_constants.USERNAME
        self.postgresql_password: str = push_data_constants.PASSWORD
        self.postgresql_port: int = push_data_constants.PORT_ID



class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_FOLDER_NAME
        )
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FOLDER_NAME, training_pipeline.DRIFT_REPORT_FILE_NAME
        )
        self.validated_train_file_path: str = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALIDATED_FOLDER_NAME, training_pipeline.TRAIN_FILE_NAME
        )
        self.validated_test_file_path: str = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALIDATED_FOLDER_NAME, training_pipeline.TEST_FILE_NAME
            )
        

class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_TRANSFORMATION_FOLDER_NAME
        )
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_FOLDER_NAME, training_pipeline.TRANSFORMED_TRAIN_FILE_NAME
        )
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_FOLDER_NAME, training_pipeline.TRANSFORMED_TEST_FILE_NAME
        )
        self.impute_obj_file_path: str = os.path.join(
            self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMATION_OBJECTS_FOLDERNAME, training_pipeline.DATA_TRANSFORMATION_IMPUTER_OBJ_FOLDER_NAME
        )
        self.target_encoding_file_path: str = os.path.join(
            self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMATION_OBJECTS_FOLDERNAME, training_pipeline.DATA_TRANSFORMATION_ENCODING_OBJ_FOLDER_NAME, training_pipeline.TARGET_ENCODING_FILE_NAME
        )
        self.ordinal_encoding_file_path: str = os.path.join(
            self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMATION_OBJECTS_FOLDERNAME, training_pipeline.DATA_TRANSFORMATION_ENCODING_OBJ_FOLDER_NAME, training_pipeline.ORDINAL_ENCODING_FILE_NAME
        )
        self.label_encoding_file_path: str = os.path.join(
            self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMATION_OBJECTS_FOLDERNAME, training_pipeline.DATA_TRANSFORMATION_ENCODING_OBJ_FOLDER_NAME, training_pipeline.LABEL_ENCODING_FILE_NAME
        )
        self.ohe_file_path: str = os.path.join(
            self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMATION_OBJECTS_FOLDERNAME, training_pipeline.DATA_TRANSFORMATION_ENCODING_OBJ_FOLDER_NAME, training_pipeline.OHE_FILE_NAME
        )
        self.scaler_file_path: str = os.path.join(
            self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMATION_OBJECTS_FOLDERNAME, training_pipeline.DATA_TRANSFORMATION_SCALER_OBJ_FOLDER_NAME, training_pipeline.SCALER_FILE_NAME
        )
        self.X_file_path: str = os.path.join(
            self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_FOLDER_NAME, training_pipeline.X_FILE_NAME
        )
        self.y_file_path: str = os.path.join(
            self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_FOLDER_NAME, training_pipeline.Y_FILE_NAME
        )
        self.high_corr_cols: list = training_pipeline.HIGH_CORR_COLS
        self.features_to_scale: list = training_pipeline.FEATURES_TO_SCALE
        self.features_to_log_transform = training_pipeline.FEATURES_TO_LOG_TRANSFORM


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINER_FOLDER_NAME
        )
        self.best_model_file_path: str = os.path.join(
            self.model_trainer_dir, training_pipeline.MODEL_TRAINER_BEST_MODEL_FOLDER_NAME, training_pipeline.BEST_MODEL_FILE_NAME
        )


class PredictionPipelineConfig:
    def __init__(self):
        self.training_pipeline_artifact_file_path: str = os.path.join(
            prediction_pipeline.TRAINING_PIPELINE_ARTIFACTS_FOLDER_NAME, prediction_pipeline.TRAINING_PIPELINE_ARTIFACTS_FILE_NAME
        )



