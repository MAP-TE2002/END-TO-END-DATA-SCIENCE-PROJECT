# END-TO-END-DATA-SCIENCE-PROJECT
COMPANY : CODTECH IT SOLUTIONS
NAME : MANDRAJULA ARUN PRABHU TEJA
INTERN ID : CT06DF375
DOMAIN : DATA SCIENCE
DURATION : 6 WEEKS
MENTOR : NEELA SANTHOSH KUMAR

🫀 Heart Disease Prediction – End-to-End Data Science Project with Flask Deployment:

This repository presents a full-stack Data Science project pipeline for predicting the likelihood of heart disease using a variety of health and demographic features. It encompasses all the stages from data collection and preprocessing to exploratory data analysis (EDA), model building, and ultimately, deployment using Flask as an API backend.

🚀 Project Overview:

Heart disease is one of the leading causes of death globally. Early detection and prediction can significantly reduce mortality rates by enabling timely medical interventions. The purpose of this project is to develop a machine learning model that can predict whether a patient is at risk of heart disease based on medical parameters such as age, blood pressure, cholesterol levels, and more.

The project is built in Python and includes the following stages:

Data understanding and exploration

Feature preprocessing and transformation

Exploratory data analysis (EDA) with visualizations

Model training using Random Forest

API development with Flask

Web-based form submission using HTML + JavaScript

Local deployment and testing

📊 Dataset:

The dataset used is heart.csv, a publicly available dataset commonly used in heart disease prediction projects. It contains the following features:

Numerical features: age, trestbps, chol, thalach, oldpeak

Categorical features: sex, cp, fbs, restecg, exang, slope, ca, thal

Target: target (0 = No Heart Disease, 1 = Heart Disease)

🧪 Model Building:

The dataset was split into training and testing sets using train_test_split. Features were scaled using StandardScaler to improve model performance. A Random Forest Classifier was chosen due to its robustness and ability to handle both categorical and numerical data.

The model was trained and evaluated, and then serialized using joblib for deployment purposes.

🔧 Model Deployment with Flask:

A lightweight Flask API was built to serve the trained model. The API includes the following key functionality:

A /predict endpoint that receives patient data via POST requests (JSON format)

Scales the input using the saved StandardScaler

Applies the trained model to generate predictions

Returns the prediction in JSON format

The backend was tested using both browser-based HTML form submissions and tools like curl or Postman.

🌐 Frontend Integration (HTML Form):

A simple HTML page (form.html) was created to allow user interaction with the API. Users can input values for all 13 model features, and upon submission, a JavaScript handler sends the data to the Flask API. The returned prediction is then displayed on the same page.


📁 Project Structure

📁 heart-disease-prediction

├── heart.csv                    # Dataset

├── retrain_model.py             # Script to preprocess data and train model

├── app.py                       # Flask application

├── heart_model.pkl              # Trained RandomForest model

├── scaler.pkl                   # Fitted StandardScaler

├── form.html                    # HTML frontend for user input

└── README.md                    # Project documentation




⚙️ Requirements:
includes:

Flask

scikit-learn

pandas

numpy

seaborn

matplotlib

plotly

joblib

📢 Conclusion:

This project demonstrates how a machine learning model can be developed and deployed for real-world use cases using Python’s data science stack and Flask. It brings together data analysis, model building, and practical deployment — a critical combination for any aspiring data scientist or ML engineer.

















































