# 7488_Churning_Customers
AI Assignment 3
Customer Churn Prediction Deployment
This project involves predicting customer churn using a machine learning model. The deployed application allows users to input customer information, and the model predicts whether the customer is likely to churn or not. Additionally, a confidence level is provided to indicate the certainty of the prediction.

Project Overview
My project included the following key steps:

Data Loading and Preprocessing: I loaded the dataset (CustomerChurn_dataset.csv) , removed irrelevant columns, handled missing values, and encoded categorical variables.

Feature Selection:I determined feature importance using a Random Forest Regressor, and selected features above a threshold of 0.02 for the model.

Model Training: I trained a machine learning model, based on a Multi-Layer Perceptron (MLP) neural network, using the selected features.

Hyperparameter Tuning: I performed grid search to optimize the hyperparameters of the model.

Model Evaluation: I evaluated the final model on a test set, and calculated key metrics such as accuracy and AUC score.

Model Deployment: I saved the trained model along with a StandardScaler, and created a Streamlit web application for user interaction.


Interact with the App: Open your web browser and go to the provided URL to interact with the application. Enter customer information, click the "Predict" button, and view the churn prediction along with the confidence level.

Application Code Overview
The application code includes:

Model Loading: The pre-trained machine learning model (best_model.h5) and the scaler (scaled.pkl) are loaded.

Input Preprocessing: The Streamlit app preprocesses user input, one-hot encodes categorical variables, and converts the input to the format expected by the model.

Prediction: The model predicts the likelihood of churn based on user input, and the result is displayed along with a confidence level.

link to video: https://youtu.be/I0WTfzibvso






