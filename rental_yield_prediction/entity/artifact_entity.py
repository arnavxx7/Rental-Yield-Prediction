from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str
    new_data_file_path: str 

@dataclass
class DataValidationArtifact:
    train_validated_status: bool
    test_validated_status: bool
    drift_status: bool
    drift_report_file_path: str
    validated_train_file_path: str
    validated_test_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    X_file_path: str
    y_file_path: str
    impute_area_obj_file_path: str
    impute_age_obj_file_path: str
    target_encoding_file_path: str
    ordinal_encoding_file_path: str
    label_encoding_file_path: str
    ohe_encoding_file_path: str
    scaler_file_path: str


@dataclass
class MetricArtifact:
    r2_score: float
    mae: float

@dataclass
class CVMetricArtifact:
    r2_score: float
    mae: float


@dataclass
class ModelTrainerArtifact:
    best_model_file_path: str
    train_metrics: MetricArtifact
    test_metrics: MetricArtifact
    cv_metrics: CVMetricArtifact

@dataclass
class TrainingPipelineArtifact:
    data_ingestion_artifact: DataIngestionArtifact
    data_validation_artifact: DataValidationArtifact
    data_transformation_artifact: DataTransformationArtifact
    model_trainer_artifact: ModelTrainerArtifact