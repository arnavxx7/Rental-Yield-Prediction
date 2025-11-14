# from rental_yield_prediction.constants.training_pipeline import SCHEMA_FILE_PATH
# from rental_yield_prediction.utils.main_utils.utils import read_yaml, load_object
# from rental_yield_prediction.entity.config_entity import PredictionPipelineConfig
import pandas as pd
import numpy as np

# data_schema = read_yaml(SCHEMA_FILE_PATH)

# df = pd.read_csv("Artifacts\\08_11_2025_15_41_49\\data_validation\\validated\\Test.csv")
# print(sum(df.isnull().sum().to_list()))

# prediction_pipeline_config = PredictionPipelineConfig()


# training_pipeline_artifact = load_object(prediction_pipeline_config.training_pipeline_artifact_file_path)

# print(training_pipeline_artifact.model_trainer_artifact.test_metrics.r2_score*100, "%")

df = pd.read_csv("New_data.csv")

# print(df.loc[0,:])
# imputer_age = load_object("Artifacts\\11_11_2025_14_24_15\\data_transformation\\transformation_objects\\imputer_obj\\imputer_age.pkl")
# df2[["age"]] = imputer_age.transform(df2[["age"]])
print(df["Veg/Non-veg"].unique())