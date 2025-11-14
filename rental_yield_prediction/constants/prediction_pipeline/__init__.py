from datetime import datetime

COLS_SELECTED = ['Suburb',
 'Floor',
 'Total_floor',
 'Parking',
 'num_BHK',
 'Furnishing',
 'Veg/Non-veg',
 'num_amenities',
 'age',
 'balconies',
 'area',
 'Family',
 'Family/Bachelor'
 ]

FEATURES_TO_SCALE = [
    "Floor",
    "Total_floor",
    "Parking",
    "num_BHK",
    "num_amenities",
    "age",
    "balconies",
    "area"
]

FEATURES_TO_LOG_TRANSFORM = [
    "area",
    "Suburb"
]
TRAINING_PIPELINE_ARTIFACTS_FOLDER_NAME: str = "Training_Pipeline_Artifacts"
TRAINING_PIPELINE_ARTIFACTS_FILE_NAME: str = "training_pipeline_artifacts.pkl"