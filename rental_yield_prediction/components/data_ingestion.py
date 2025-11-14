import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, URL
from rental_yield_prediction.entity.config_entity import DataIngestionConfig
from rental_yield_prediction.entity.artifact_entity import DataIngestionArtifact
from rental_yield_prediction.exception.exception import CustomException
from rental_yield_prediction.logging.logger import logging
from sklearn.model_selection import train_test_split

class DataIngestion:
    '''
    Extract data from sql server as a dataframe, 
    Split the data
    Save the raw data and the splitted train, test, new data
    Returns file paths for train, test and new data
    '''
    def __init__(self, dataingestionconfig:DataIngestionConfig):
        try:
            self.data_ingestion_config = dataingestionconfig
        except Exception as e:
            raise CustomException(e, sys)
    
    def extract_data_as_dataframe(self, url_obj) -> pd.DataFrame:
        try:
            engine = create_engine(url_obj)
            raw_data = pd.read_sql(sql="rental_data", con=engine, index_col="id")
            logging.info("Data extracted from sql server into dataframe")
            return raw_data
        except Exception as e:
            raise CustomException(e, sys)
    def export_data_to_feature_store(self, raw_data: pd.DataFrame):
        try:
            dir_path = os.path.dirname(self.data_ingestion_config.raw_data_file_path)
            os.makedirs(dir_path, exist_ok=True)
            raw_data.to_csv(
                self.data_ingestion_config.raw_data_file_path, index=False, header=True
            )
            logging.info("Raw data stored in feature store folder")
            return raw_data
        except Exception as e:
            raise CustomException(e, sys)
    
    def split_data(self, raw_data: pd.DataFrame):
        try:
            training_data, new_data = train_test_split(raw_data, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42)
            train_data, test_data = train_test_split(training_data, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42)

            dir_path = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_data.to_csv(
                self.data_ingestion_config.train_file_path, index=False, header=True
            )
            test_data.to_csv(
                self.data_ingestion_config.test_file_path, index=False, header=True
            )
            new_data.to_csv(
                self.data_ingestion_config.new_data_file_path, index=False, header=True
            )
            logging.info("Data splitted successfully into train, test, new data")
            logging.info("All 3 dataframes stored in ingested folder")
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion initiated")
            username = self.data_ingestion_config.postgresql_username
            password = self.data_ingestion_config.postgresql_password
            host = self.data_ingestion_config.postgresql_hostname
            database = self.data_ingestion_config.postgresql_database

            url_obj = URL.create(
                "postgresql",
                username=username,
                password=password,
                host=host,
                database=database,
            )

            raw_data = self.extract_data_as_dataframe(url_obj)
            raw_data = self.export_data_to_feature_store(raw_data)
            self.split_data(raw_data)
            dataingestionartifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path,
                new_data_file_path=self.data_ingestion_config.new_data_file_path
            )
            logging.info("Data Ingestion completed")
            return dataingestionartifact
        except Exception as e:
            raise CustomException(e, sys)


