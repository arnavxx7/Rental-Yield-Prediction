import pandas as pd
import numpy as np
import os
import sys
import mlflow
import dagshub
from rental_yield_prediction.exception.exception import CustomException
from rental_yield_prediction.logging.logger import logging
from rental_yield_prediction.entity.config_entity import ModelTrainerConfig
from rental_yield_prediction.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from rental_yield_prediction.utils.main_utils.utils import load_numpy_array_data, evaluate_model, save_object
from rental_yield_prediction.utils.ml_utils.metrics import get_metrics, get_cv_metrics

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor



class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise CustomException(e, sys)
        
    def track_mlflow(self, best_model, metrics, metric_type: str):
        try:
            with mlflow.start_run():
                r2_score = metrics.r2_score
                mae = metrics.mae
                mlflow.set_tag("Metric Type", metric_type)
                mlflow.log_param("R2 Score", r2_score)
                mlflow.log_param("MAE", mae)
                mlflow.sklearn.log_model(best_model,"model")
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_best_model(self, X, y, X_train, X_test, y_train, y_test):
        try:
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Ada Boost": AdaBoostRegressor(),
                "Xgboost": XGBRegressor()
            }
            params = {
                "Linear Regression": {
                    "tol":[np.float_(1e-6), np.float_(1e-5), np.float_(1e-7)]
                },
                "Ridge": {
                    "alpha": [1, 0.8, 1.2]
                },
                "Lasso": {
                    "alpha": [1, 0.8, 1.2]
                },
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    "max_depth": [None, 50, 100],
                    "min_samples_leaf": [1, 5, 10]
                },
                "Random Forest": {
                    "n_estimators": [100, 150, 300],
                    "min_samples_split": [2, 5, 10],
                    "max_features": ["sqrt", "log2", 2, 1]
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 150, 300],
                    "min_samples_split": [2, 5, 10],
                    "max_features": ["sqrt", "log2", 2, 1]
                },
                "Ada Boost": {
                    "n_estimators": [50, 100, 150],
                    "learning_rate": [1, 0.8, 1.2]
                },
                "Xgboost": {
                    "learning_rate": [0.3, 0.5, 0.7],
                    "max_depth": [5,6,7]
                }
            }
            model_report, best_params = evaluate_model(X, y, X_train, X_test, y_train, y_test, models, params)
            best_score = max(list(model_report.values()))
            best_model_idx = list(model_report.values()).index(best_score)
            best_model_name = list(model_report.keys())[best_model_idx]
            best_model = models[best_model_name]
            best_param = best_params[best_model_name]
            logging.info(f"{best_model_name} is the best model with a r2 score of {best_score}")
            logging.info(f"{best_model_name} has the best parameters - {best_param}")
            return best_model, best_param
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_model_training(self):
        try:
            logging.info("Model Training Initiated")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #read the train and test arrays
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            X = pd.read_csv(self.data_transformation_artifact.X_file_path)
            y = pd.read_csv(self.data_transformation_artifact.y_file_path)
            logging.info(f"X has been read - {X.shape}")
            logging.info(f"y has been read - {y.shape}")
            logging.info(f"X_train, X_test, y_train, y_test shape = [{X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}]")
            best_model, best_param = self.get_best_model(X, y, X_train, X_test, y_train, y_test)
            best_model.set_params(**best_param)
            best_model.fit(X_train, y_train)
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)
            train_metrics = get_metrics(y_train, y_pred_train)
            test_metrics = get_metrics(y_test, y_pred_test)
            cv_metrics = get_cv_metrics(best_model, X, y)
            # Log metrics in mlflow
            # self.track_mlflow(best_model, train_metrics, "Train")
            # self.track_mlflow(best_model, test_metrics, "Test")
            # self.track_mlflow(best_model, cv_metrics, "CV")
            print("\n")
            print(f"Best Model - {best_model}")
            print("Train metrics - ")
            print(train_metrics.r2_score)
            print(train_metrics.mae)
            print("Test metrics - ")
            print(test_metrics.r2_score)
            print(test_metrics.mae)
            print("CV Metrics - ")
            print(cv_metrics.r2_score)
            print(cv_metrics.mae)
            logging.info(f"{best_model} has been fit on the training data, and set on the best params")
            save_object(self.model_trainer_config.best_model_file_path, best_model)
            logging.info("Trained best model saved")
            model_trainer_artifact = ModelTrainerArtifact(
                best_model_file_path=self.model_trainer_config.best_model_file_path,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                cv_metrics=cv_metrics
            )
            logging.info("Model Training completed")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)
            
