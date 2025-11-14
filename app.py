from rental_yield_prediction.pipeline.prediction_pipeline import PredictionPipeline
from rental_yield_prediction.pipeline.training_pipeline import TrainingPipeline
from rental_yield_prediction.exception.exception import CustomException
from rental_yield_prediction.utils.main_utils.utils import load_object
import pandas as pd
import numpy as np
import sys
import uvicorn
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)  
from fastapi import FastAPI, Request, Form
from starlette.responses import RedirectResponse




app  = FastAPI()

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates")

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def start_training_pipeline(req: Request):
    '''
    Starts training pipeline by ingesting data from PostgreSQL server, 
    followed by data validation, transformation, and model training.
    '''
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline_artifact = training_pipeline.execute_training_pipeline()
        best_model = load_object(training_pipeline_artifact.model_trainer_artifact.best_model_file_path)
        cv_metrics = {"r2": round(training_pipeline_artifact.model_trainer_artifact.cv_metrics.r2_score*100, 3), "mae": round(np.exp(training_pipeline_artifact.model_trainer_artifact.cv_metrics.mae), 3)}
        train_metrics = {"r2": round(training_pipeline_artifact.model_trainer_artifact.train_metrics.r2_score*100, 3), "mae": round(np.exp(training_pipeline_artifact.model_trainer_artifact.train_metrics.mae), 3)}
        test_metrics = {"r2": round(training_pipeline_artifact.model_trainer_artifact.test_metrics.r2_score*100, 3), "mae": round(np.exp(training_pipeline_artifact.model_trainer_artifact.test_metrics.mae), 3)}
        return templates.TemplateResponse(
            name="training_page.html",
            context={
                "request": req,
                "cv_metrics": cv_metrics,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "best_model": best_model
            }
        )
    except Exception as e:
        raise CustomException(e, sys)
    

@app.get("/predict")
async def display_form(req: Request):
    return templates.TemplateResponse(
        name="predict_page.html",
        context={
            "request": req,
        }
    )

@app.post("/predict")
async def start_prediction_pipeline(
    req: Request,
    suburb: str = Form(...),
    floor: int = Form(...),
    total_floors: int = Form(...),
    parking_spaces: int = Form(...),
    bhk: int = Form(...),
    pref: str = Form(...),
    furnishing_status: str = Form(...),
    food_pref: str = Form(...),
    num_amenities: int = Form(...),
    building_age: int = Form(...),
    num_balconies: int = Form(...),
    area_sqft: float = Form(...),
    budget: int = Form(...)    
):
    if area_sqft is None:
       area_sqft = np.nan

    
    if building_age is None:
        building_age = np.nan


    input_dict = {
        "Suburb": [suburb],
        "Floor": [floor],
        "Total_floor": [total_floors],
        "Parking": [parking_spaces],
        "num_BHK": [bhk], 
        "Preference": [pref],
        "Furnishing": [furnishing_status],
        "Veg/Non-veg": [food_pref],
        "num_amenities": [num_amenities],
        "age": [building_age],
        "balconies": [num_balconies],
        "area": [area_sqft]
    }
    df = pd.DataFrame(input_dict) 
    df = df.loc[[0], :]
    prediction_pipeline = PredictionPipeline(df)
    predicted_rent = prediction_pipeline.execute_prediction_pipeline()
    annual_rent_income = predicted_rent*12
    predicted_yield = (annual_rent_income/budget)*100
    return templates.TemplateResponse(
        name="predict_page.html",
        context={
            "request": req,
            "rent_prediction": round(predicted_rent, 3),
            "yield_prediction": round(predicted_yield, 3)
        }
   )



if __name__=="__main__":
    uvicorn.run("app:app", reload=True)   

