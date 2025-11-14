
import pandas as pd
# Source - https://stackoverflow.com/a
# Posted by astrofrog, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-10, License - CC BY-SA 4.0

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os
import sys
from rental_yield_prediction.entity.config_entity import DataTransformationConfig
from rental_yield_prediction.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from rental_yield_prediction.exception.exception import CustomException
from rental_yield_prediction.logging.logger import logging
from rental_yield_prediction.utils.main_utils.utils import save_numpy_array_data, save_object
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, OrdinalEncoder, LabelEncoder, StandardScaler

class DataTransformation:
    '''
    Reads the train, test data
    Performs imputation, encodes categorical features
    Saves transformed train, test as numpy array
    Saves transformation objects eg - impute, te, ohe, etc as pkl files
    Returns file paths for all saved files
    '''
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_valiation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomException(e, sys)
        
    def impute_missing_vals(self, train_df: pd.DataFrame, test_df: pd.DataFrame, col: str):
        try:
            knnimpute = KNNImputer(n_neighbors=5, weights="uniform")
            train_df[[col]] = knnimpute.fit_transform(train_df[[col]])
            test_df[[col]] = knnimpute.transform(test_df[[col]])
            return train_df, test_df, knnimpute
        except Exception as e:
            logging.error(f"Error in imputation - {e}")
            raise CustomException(e, sys)
        
    def ode_feature(self, train_df: pd.DataFrame, test_df: pd.DataFrame, col: str):
        try:
            ode = OrdinalEncoder(categories=[["UnFurnished", "Semi Furnished", "Fully Furnished"]])
            train_df[[col]] = ode.fit_transform(train_df[[col]])
            test_df[[col]] = ode.transform(test_df[[col]])
            return train_df, test_df, ode
        except Exception as e:
            logging.error(f"Error in encoding {ode}, error - {e}")
            raise CustomException(e, sys)
        
    def le_feature(self, train_df: pd.DataFrame, test_df: pd.DataFrame, col: str):
        try:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col])
            test_df[col] = le.transform(test_df[col])
            return train_df, test_df, le
        except Exception as e:
            logging.error(f"Error in {le}, error - {e}")
            raise CustomException(e, sys)
        
        
    def ohe_feature(self, train_df: pd.DataFrame, test_df: pd.DataFrame, col: str):
        try:
            ohe = OneHotEncoder()
            pref = ohe.fit_transform(train_df[[col]])
            pref_df =  pd.DataFrame(pref.toarray(), columns=ohe.categories_[0])
            train_df = pd.concat([train_df, pref_df], axis=1)
            train_df.drop(columns=col, inplace=True)
            pref2 = ohe.transform(test_df[[col]])
            pref_df2 = pd.DataFrame(pref2.toarray(), columns=ohe.categories_[0])
            test_df = pd.concat([test_df, pref_df2], axis=1)
            test_df.drop(columns=col, inplace=True)
            return train_df, test_df, ohe
        except Exception as e:
            logging.error(f"Error in ohe - {e}")
            raise CustomException(e, sys)
    
    def te_feature(self, train_df: pd.DataFrame, test_df: pd.DataFrame, col: str):
        try:
            te = TargetEncoder()
            train_df[[col]] = te.fit_transform(train_df[[col]], train_df["Rent"])
            test_df[[col]] = te.transform(test_df[[col]])
            return train_df, test_df, te
        except Exception as e:
            logging.error(f"Error in target encoding, error is - {e}")
            raise CustomException(e, sys)
        
    def log_transform(self, dataframe, features):
        try:
            dataframe.loc[:, features] = np.log(dataframe.loc[:, features])
            return dataframe
        except Exception as e:
            logging.error(f"Error in log transformation - {e}")
            raise CustomException(e, sys)
        
    def scale_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame, features_to_scale: list):
        try:
            scaler = StandardScaler()
            train_df.loc[:, features_to_scale] = scaler.fit_transform(train_df.loc[:, features_to_scale])
            test_df.loc[:, features_to_scale] = scaler.transform(test_df.loc[:, features_to_scale])
            return train_df, test_df, scaler
        except Exception as e:
            logging.error(f"Error in scaling - {e}")
            raise CustomException(e, sys)
    
    @staticmethod
    def read_file(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            # Read the train and test files
            logging.info("Initiating Data Transformation")
            train_df = DataTransformation.read_file(file_path=self.data_valiation_artifact.validated_train_file_path)
            test_df = DataTransformation.read_file(file_path=self.data_valiation_artifact.validated_test_file_path)
            logging.info("Read the train and test dataframes")
            # Reset the index
            train_df.reset_index(drop=True, inplace=True)
            test_df.reset_index(drop=True, inplace=True)
            logging.info(f"Train dataframe shape = {train_df.shape}")
            logging.info(f"Test dataframe shape = {test_df.shape}")
            #Imputing missing values
            train_df, test_df, imputer_obj_age = self.impute_missing_vals(train_df=train_df, test_df=test_df, col="age")
            train_df, test_df, imputer_obj_area = self.impute_missing_vals(train_df=train_df, test_df=test_df, col="area")
            logging.info("Missing values in age and area column imputed")
            logging.info(f"Train dataframe missing values = {sum(train_df.isnull().sum().to_list())}")
            logging.info(f"Test dataframe missing values = {sum(test_df.isnull().sum().to_list())}")
            #Encoding categorical features
            train_df, test_df, target_enc = self.te_feature(train_df=train_df, test_df=test_df, col="Suburb")
            logging.info(f"Suburb column target encoded, Train = {train_df["Suburb"].mean()}, Test = {test_df["Suburb"].mean()}")
            train_df, test_df, ordinal_enc = self.ode_feature(train_df=train_df, test_df=test_df, col="Furnishing")
            logging.info(f"Furnishing column ordinal encoded {train_df["Furnishing"].unique().tolist()}")
            train_df, test_df, label_enc = self.le_feature(train_df=train_df, test_df=test_df, col="Veg/Non-veg")
            logging.info(f"Veg/Non-veg column label encoded {train_df["Veg/Non-veg"].unique().tolist()}")
            train_df, test_df, ohe = self.ohe_feature(train_df=train_df, test_df=test_df, col="Preference")
            logging.info(f"Preference column one hot encoded, {train_df.shape, test_df.shape}")
            # Log transform select feartures
            train_df = self.log_transform(dataframe=train_df, features=self.data_transformation_config.features_to_log_transform)
            test_df = self.log_transform(dataframe=test_df, features=self.data_transformation_config.features_to_log_transform)
            logging.info(f"{self.data_transformation_config.features_to_log_transform} log transformed")
            #Scale the features
            train_df, test_df, scaler = self.scale_features(train_df=train_df, test_df=test_df, features_to_scale=self.data_transformation_config.features_to_scale)
            logging.info(f"{self.data_transformation_config.features_to_scale} scaled")
            # Selecting only high corr features
            train_df = train_df[self.data_transformation_config.high_corr_cols]
            test_df = test_df[self.data_transformation_config.high_corr_cols]
            logging.info(f"Train shape after transformation = {train_df.shape}")
            logging.info(f"Test shape after transformation = {test_df.shape}")
            full_data_df = pd.concat([train_df, test_df], axis=0)
            full_data_df.reset_index(drop=True, inplace=True)
            X = full_data_df.iloc[:, :-1]
            y = full_data_df.iloc[:, -1]
            logging.info(f"X dataframe shape = {X.shape}")
            logging.info(f"y dataframe shape = {y.shape}")
            # Saving the transformed data and transformation objects
            imputer_obj_age_file_path = os.path.join(self.data_transformation_config.impute_obj_file_path, "imputer_age.pkl")
            imputer_obj_area_file_path = os.path.join(self.data_transformation_config.impute_obj_file_path, "imputer_area.pkl")
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_file_path, array=train_df)
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_file_path, array=test_df)
            X.to_csv(self.data_transformation_config.X_file_path, index=False, header=True)
            y.to_csv(self.data_transformation_config.y_file_path, index=False, header=True)
            save_object(file_path=imputer_obj_age_file_path, obj=imputer_obj_age)
            save_object(file_path=imputer_obj_area_file_path, obj=imputer_obj_area)
            save_object(file_path=self.data_transformation_config.target_encoding_file_path, obj=target_enc)
            save_object(file_path=self.data_transformation_config.ordinal_encoding_file_path, obj=ordinal_enc)
            save_object(file_path=self.data_transformation_config.label_encoding_file_path, obj=label_enc)
            save_object(file_path=self.data_transformation_config.ohe_file_path, obj=ohe)
            save_object(file_path=self.data_transformation_config.scaler_file_path, obj=scaler)
            logging.info("Saved all files and objects")

            # Returning the data transformation artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                X_file_path=self.data_transformation_config.X_file_path,
                y_file_path=self.data_transformation_config.y_file_path,
                impute_age_obj_file_path=imputer_obj_age_file_path,
                impute_area_obj_file_path=imputer_obj_area_file_path,
                target_encoding_file_path=self.data_transformation_config.target_encoding_file_path,
                ordinal_encoding_file_path=self.data_transformation_config.ordinal_encoding_file_path,
                label_encoding_file_path=self.data_transformation_config.label_encoding_file_path,
                ohe_encoding_file_path=self.data_transformation_config.ohe_file_path,
                scaler_file_path=self.data_transformation_config.scaler_file_path
            )
            logging.info("Data Transformation completed")

            return data_transformation_artifact
        except Exception as e:
            logging.error(f"Error ocurrend while running data transformation - {e}")
            raise CustomException(e, sys)




