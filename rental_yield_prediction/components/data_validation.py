import pandas as pd
import numpy as np
import os
import sys
from scipy.stats import ks_2samp
from rental_yield_prediction.entity.config_entity import DataValidationConfig
from rental_yield_prediction.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from rental_yield_prediction.exception.exception import CustomException
from rental_yield_prediction.logging.logger import logging
from rental_yield_prediction.constants.training_pipeline import SCHEMA_FILE_PATH
from rental_yield_prediction.utils.main_utils.utils import read_yaml, write_yaml_file

class DataValidation:
    '''
    Reads the train, test data. \n
    Checks the number of columns in each data with the data schema. \n
    Outputs the result of check as boolean. \n
    Checks for data drift. \n
    Outputs the results as data drift report. \n
    Saves report, validated train, test data. \n
    Returns check status, drift report file path, validated train and test file path.
    '''
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)
    
    def validate_data(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(list(self._schema_config.values())[0])
            logging.info(f"Number of columns = {number_of_columns}")
            logging.info(f"Number of columns in dataframe = {len(dataframe.columns)}")
            if len(dataframe.columns)==number_of_columns:
                return True
            else:
                return False

        except Exception as e:
            raise CustomException(e, sys)
        

    def detect_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05)->bool:
        try:
            report = {}
            status = True
            for col in base_df.columns:
                d1 = base_df[col]
                d2 = current_df[col]
                is_same_dist = ks_2samp(d1, d2)
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                    status=False
                report.update({col:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                    }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            #Create directory for drift report
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            # Saving drift report
            write_yaml_file(file_path=drift_report_file_path,content=report)
            return status
        except Exception as e:
            raise CustomException(e, sys)
        
    @staticmethod
    def read_file(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_validation(self)-> DataValidationArtifact:
        try:
            logging.info("Data Validation Initiated")
            # Read train and test dataframes
            train_data = self.read_file(self.data_ingestion_artifact.train_file_path)
            test_data = self.read_file(self.data_ingestion_artifact.test_file_path)
            # Validate the number of columns
            train_status = self.validate_data(train_data)
            if not train_status:
                logging.error("Train dataframe does not have all columns")
            test_status = self.validate_data(test_data)
            if not test_status:
                logging.error("Test dataframe does not have all columns")
            # Check for data drift
            drift_status = self.detect_drift(train_data, test_data)
            logging.info(f"Drift detected status = {drift_status}")

            # Create directory for validation artifacts
            dir_path = os.path.dirname(self.data_validation_config.validated_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Saving files
            train_data.to_csv(self.data_validation_config.validated_train_file_path, index=False, header=True)
            test_data.to_csv(self.data_validation_config.validated_test_file_path, index=False, header=True)
            logging.info("Train, Test, drift report saved")

            # Returning the output as artifact
            data_validation_artifact = DataValidationArtifact(
                train_validated_status=train_status,
                test_validated_status=test_status,
                drift_status=drift_status,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
                validated_train_file_path=self.data_validation_config.validated_train_file_path,
                validated_test_file_path=self.data_validation_config.validated_test_file_path
            )
            logging.info("Data Validation completed")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys)

