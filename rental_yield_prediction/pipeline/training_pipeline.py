from rental_yield_prediction.components.data_ingestion import DataIngestion
from rental_yield_prediction.components.data_validation import DataValidation
from rental_yield_prediction.components.data_transformation import DataTransformation
from rental_yield_prediction.components.model_training import ModelTrainer
from rental_yield_prediction.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, PredictionPipelineConfig
from rental_yield_prediction.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact, TrainingPipelineArtifact
from rental_yield_prediction.utils.main_utils.utils import save_object
from rental_yield_prediction.logging.logger import logging


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.prediction_pipeline_config = PredictionPipelineConfig()

    def start_data_ingestion(self):
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        return data_ingestion_artifact
    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
        return data_validation_artifact
    
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        return data_transformation_artifact
    
    def start_model_training(self, data_transformation_artifact: DataTransformationArtifact):
        model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)
        model_trainer = ModelTrainer(data_transformation_artifact, model_trainer_config)
        model_trainer_artifact = model_trainer.initiate_model_training()
        return model_trainer_artifact

    def execute_training_pipeline(self)->TrainingPipelineArtifact:
        logging.info("Training Pipeline Inititated")
        data_ingestion_artifact = self.start_data_ingestion()
        print(data_ingestion_artifact.test_file_path)
        data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
        print(data_validation_artifact.drift_status)
        data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
        print(data_transformation_artifact.impute_age_obj_file_path)
        model_trainer_artifact = self.start_model_training(data_transformation_artifact)
        print(model_trainer_artifact.cv_metrics.r2_score*100)
        training_pipeline_artifact = TrainingPipelineArtifact(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_artifact=data_validation_artifact,
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_artifact=model_trainer_artifact
        )
        save_object(file_path=self.prediction_pipeline_config.training_pipeline_artifact_file_path, obj=training_pipeline_artifact)
        logging.info("All artifacts for current training pipeline run saved")
        logging.info("Training Pipeline Completed")
        return training_pipeline_artifact


