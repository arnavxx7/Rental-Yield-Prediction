import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

from rental_yield_prediction.utils.main_utils.utils import load_object
from rental_yield_prediction.exception.exception import CustomException
from rental_yield_prediction.logging.logger import logging
from rental_yield_prediction.entity.config_entity import PredictionPipelineConfig
from rental_yield_prediction.constants.prediction_pipeline import FEATURES_TO_LOG_TRANSFORM, FEATURES_TO_SCALE, COLS_SELECTED

class PredictionPipeline:
    def __init__(self, dataframe:pd.DataFrame):
        self.input_df: pd.DataFrame = dataframe
        self.prediction_pipeline_config = PredictionPipelineConfig()
        self.training_pipeline_artifact = load_object(self.prediction_pipeline_config.training_pipeline_artifact_file_path)

    def impute_missing(self, col: str, file_path: str):
        try:
            imputer_obj = load_object(file_path)

            self.input_df[[col]] = imputer_obj.transform(self.input_df[[col]])
            logging.info(f"Missing values imputed for {col} using {imputer_obj} imputer")
            return self.input_df
        except Exception as e:
            raise CustomException(e, sys)
        
    def ode_feature(self, col: str, file_path: str):
        try:
            ode_obj = load_object(file_path)
            self.input_df[[col]] = ode_obj.transform(self.input_df[[col]])
            logging.info(f"Feature {col} ordinal encoded using {ode_obj} encoder")
            return self.input_df
        except Exception as e:
            raise CustomException(e, sys)
        
    def le_feature(self, col: str, file_path: str):
        try:
            le_obj = load_object(file_path)
            self.input_df[col] = le_obj.transform(self.input_df[col])
            logging.info(f"Feature {col} label encoded using {le_obj} encoder")
            return self.input_df
        except Exception as e:
            raise CustomException(e, sys)
        
    def ohe_feature(self, col: str, file_path: str):
        try:
            ohe_obj = load_object(file_path)
            pref = ohe_obj.transform(self.input_df[[col]])
            pref_df = pd.DataFrame(pref.toarray(), columns=ohe_obj.categories_[0])
            self.input_df = pd.concat([self.input_df, pref_df], axis=1)
            self.input_df.drop(columns=col, inplace=True)
            logging.info(f"Feature {col} one hot encoded using {ohe_obj} encoder")
            return self.input_df
        except Exception as e:
            raise CustomException(e, sys)
        
    def te_feature(self, col: str, file_path: str):
        try:
            te_obj = load_object(file_path)
            self.input_df[[col]] = te_obj.transform(self.input_df[[col]])
            logging.info(f"Feature {col} target encoded using {te_obj} encoder")
            return self.input_df
        except Exception as e:
            raise CustomException(e, sys)

    def scale_feature(self, features: list, file_path: str):
        try:
            scaler_obj = load_object(file_path)
            self.input_df.loc[:, features] = scaler_obj.transform(self.input_df.loc[:, features])    
            logging.info(f"Features {features} scaled using {scaler_obj} object")
            return self.input_df
        except Exception as e:
            raise CustomException(e, sys)

    def execute_prediction_pipeline(self):
        try:
            logging.info("Prediction Pipleine Initiated")
            self.input_df.reset_index(drop=True, inplace=True)
            self.input_df = self.impute_missing(col="age", file_path=self.training_pipeline_artifact.data_transformation_artifact.impute_age_obj_file_path)
            self.input_df = self.impute_missing(col="area", file_path=self.training_pipeline_artifact.data_transformation_artifact.impute_area_obj_file_path)
            self.input_df = self.ode_feature(col="Furnishing", file_path=self.training_pipeline_artifact.data_transformation_artifact.ordinal_encoding_file_path)
            self.input_df = self.le_feature(col="Veg/Non-veg", file_path=self.training_pipeline_artifact.data_transformation_artifact.label_encoding_file_path)
            self.input_df = self.ohe_feature(col="Preference", file_path=self.training_pipeline_artifact.data_transformation_artifact.ohe_encoding_file_path)
            self.input_df = self.te_feature(col="Suburb", file_path=self.training_pipeline_artifact.data_transformation_artifact.target_encoding_file_path)
            self.input_df.loc[:, FEATURES_TO_LOG_TRANSFORM] = np.log(self.input_df.loc[:, FEATURES_TO_LOG_TRANSFORM])
            logging.info(f"{FEATURES_TO_LOG_TRANSFORM} log transformed using np.log")
            self.input_df = self.scale_feature(FEATURES_TO_SCALE, self.training_pipeline_artifact.data_transformation_artifact.scaler_file_path)
            self.input_df = self.input_df[COLS_SELECTED]
            logging.info(f"Shape of input dataframe after transformation - {self.input_df.shape}")
            print(self.input_df.shape)
            best_model = load_object(self.training_pipeline_artifact.model_trainer_artifact.best_model_file_path)
            y_pred = best_model.predict(self.input_df)
            logging.info(f"Prediction = {np.exp(y_pred.mean())}")
            logging.info("Prediction Pipeline completed")
            return np.exp(y_pred.mean())
        
        except Exception as e:
            raise CustomException(e, sys)



    
