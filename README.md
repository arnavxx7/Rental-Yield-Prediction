Rental Yield Prediction — End-to-End ML Project
This project is an end-to-end machine learning system designed to predict the rental yield of residential apartments in Mumbai. 
The model uses features such as the property's suburb, floor number, total floors in the building, number of parking spaces, BHK configuration, furnishing status, preferred tenant type, amenities, budget, area, and more.
Real-time data is scraped from a property listing website using BeautifulSoup and stored in a PostgreSQL database, forming the foundation for training the model.

A complete automated training pipeline is implemented, beginning with data ingestion from PostgreSQL, followed by data validation steps that ensure the correct structure of the dataset and detect potential data drift.
The data transformation stage prepares the dataset through preprocessing operations such as scaling and encoding before the model training phase begins.
Multiple models are trained with hyperparameter tuning, and the best-performing model is selected, achieving a test R² score of 77%.
This model is saved and later used for real-time predictions through a dedicated prediction pipeline.

To make the system accessible and user-friendly, a FastAPI web application is integrated with both the training and prediction pipelines.
The /train route triggers the full training workflow while displaying validation results and model performance metrics, and the /predict route provides an interface for users to input property details and receive instant rental yield predictions.
The entire experiment lifecycle, including metrics, parameters, and artifacts, is tracked using MLflow with DagsHub as the backend, allowing seamless monitoring and versioning of models.

