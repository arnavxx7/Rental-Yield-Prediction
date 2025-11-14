import yaml
from rental_yield_prediction.exception.exception import CustomException 
from rental_yield_prediction.logging.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import os,sys
import numpy as np
import pickle


def read_yaml(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys)

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e, sys)
    

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
        logging.info(f"Saved {array.shape} successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)
    
    
def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Saved {obj} successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exists")
        with open(file_path, "rb") as file_obj:
            logging.info(f"{file_obj} loaded")
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            logging.info(f"{file_obj} loaded")
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X, y, X_train, X_test, y_train, y_test, models: dict, params: dict):
    try:
        report = {}
        best_params = {}
        for model_name, model in models.items():
            logging.info(f"Training {model_name}")
            param = params[model_name]
            gs = GridSearchCV(estimator=model, param_grid=param, scoring="r2", cv=5)
            gs.fit(X, y)
            best_params[model_name] = gs.best_params_
            logging.info(f"Grid search best cross val r2 score for {model_name} = {gs.best_score_}")
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            train_r2_score = r2_score(y_train, y_pred_train)
            test_r2_score = r2_score(y_test, y_pred_test)
            logging.info(f"Train r2 score for {model_name} = {train_r2_score}")
            logging.info(f"Test r2 score for {model_name} = {test_r2_score}")
            report[model_name] = test_r2_score
        logging.info("Model report generated successfully")
        return report, best_params
    except Exception as e:
        raise CustomException(e, sys)
         

