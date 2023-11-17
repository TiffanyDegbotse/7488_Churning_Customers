import pickle
import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the model
model = load_model("best_model.h5")
print(model.input_names)

# Load the scaler
with open("scaled.pkl", "rb") as scaler_file:
    loaded_object = pickle.load(scaler_file)

# Check if the loaded object is a StandardScaler
if not isinstance(loaded_object, StandardScaler):
    raise ValueError("The loaded object is not a StandardScaler.")

scale = loaded_object

# Mapping for categorical features
contract_mapping = {"Month-to-month": 0, "One year": 1, "Two years": 2}
payment_method_mapping = {"Electronic check": 0, "Mailed check": 1, "Bank transfer": 2, "Credit card": 3}
binary_mapping = {"Yes": 1, "No": 0, "No internet service": 0}

# Function to preprocess input features
def preprocess_input(tenure, MonthlyCharges, TotalCharges, gender, Partner, OnlineSecurity, TechSupport, Contract, PaperlessBilling, PaymentMethod):
    # Map binary categorical features
    OnlineSecurity = binary_mapping.get(OnlineSecurity, 0)
    TechSupport = binary_mapping.get(TechSupport, 0)
    PaperlessBilling = binary_mapping.get(PaperlessBilling, 0)
    Partner = binary_mapping.get(Partner, 0)

    # Map other categorical features
    Contract = contract_mapping.get(Contract, 0)
    PaymentMethod = payment_method_mapping.get(PaymentMethod, 0)

    # Create a Pandas DataFrame with the input features
    input_df = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges],
        'gender': [gender],
        'Partner': [Partner],
        'OnlineSecurity': [OnlineSecurity],
        'TechSupport': [TechSupport],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
    })

    # One-hot encode categorical variables
    input_df = pd.get_dummies(input_df, columns=['gender'])

    # Convert to float32
    input_df = input_df.astype('float32')

    return np.array(input_df)

# Defining the Streamlit app
def main():
    st.title("Churning Predictor")

    # Creating input fields for the features
    st.write("Feature values for prediction:")
    monthly_charges = st.number_input("Enter Monthly Charges", value=0.0)
    total_charges = st.number_input("Enter Total Charges", value=0.0)
    tenure = st.number_input("Enter Tenure", value=0)

    # Other features can be added similarly
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two years"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    partner = st.selectbox("Partner", ["Yes", "No"])

    # Assuming you want to predict using these features
    if st.button("Predict"):
        # Preprocess input features
        input_features = preprocess_input(tenure, monthly_charges, total_charges, gender, partner, online_security, tech_support, contract,
                                          paperless_billing, payment_method)
        # Make a prediction
        prediction = model.predict(input_features)

        # Convert probabilities to class labels based on the threshold
        #Calculated the threshold by 2000/5000 100 using the EDA graph
        threshold = 0.4 # You can adjust this threshold as needed
        prediction_class = (prediction > threshold).astype(int)

        # Map class labels to 'Churn' and 'Not Churn'
        prediction_label = 'Churn' if prediction_class == 1 else 'Not Churn'

        # Display the prediction or take further actions
        st.write(f"Churning Prediction: {prediction_label}")

        # Display the prediction and confidence level
        st.write(f"Confidence factor of the model")
        st.write("77%")

if __name__ == "__main__":
    main()
