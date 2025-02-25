import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
import google.generativeai as genai
import logging

# RecoveryInfoProcessor class to retrieve recovery steps
class RecoveryInfoProcessor:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def get_recovery_info(self, issue: str) -> str:
        prompt_text = f"Please provide recovery steps for the following water quality issue: {issue}"
        try:
            response = self.model.generate_content([prompt_text])
            return response.text.strip()
        except Exception as e:
            logging.error(f"Error fetching recovery information: {str(e)}")
            return "Unable to fetch recovery information. Please try again later."

# Load the trained model
model = joblib.load('stacking_model.pkl')

# Data preprocessing function
def process_data(data):
    # Apply transformations to input data
    data['Log_Hardness'] = np.log1p(data['Hardness'])

    # Handle missing values with median imputation
    imputer = SimpleImputer(strategy='median')
    data_imputed = imputer.fit_transform(data)

    # Convert the imputed data back to a DataFrame
    data_imputed = pd.DataFrame(data_imputed, columns=data.columns)

    # Polynomial feature transformation
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(data_imputed)
    poly_feature_names = poly.get_feature_names_out(data_imputed.columns)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    
    return data_imputed, poly_df

# Health assessment function
def health_assessment(prediction):
    if prediction == 1:
        return "The water quality is good for health and safe for consumption."
    else:
        return "The water quality is bad for health and may pose health risks if consumed."

# Streamlit app
st.set_page_config(page_title="Water Quality Prediction and Recovery Steps", layout="wide")

st.title("Water Quality Prediction and Recovery Steps")

# Create a two-column layout with increased width for the right column
main_col, right_col = st.columns([2, 1])

# File uploader
with main_col:
    uploaded_file = st.file_uploader("Upload a CSV file with water quality data", type="csv")

    # Placeholder for results
    results_placeholder = st.empty()

    if uploaded_file is not None:
        try:
            # Read and display the CSV file
            data = pd.read_csv(uploaded_file)
            st.write("Input Data")
            st.dataframe(data)

            # Add a button to trigger predictions
            if st.button('Predict'):
                # Process the data
                original_data, processed_data = process_data(data)

                # Perform prediction
                predictions = model.predict(processed_data)

                # Generate health assessments
                assessments = [health_assessment(pred) for pred in predictions]

                # Display results
                result_data = original_data.copy()
                result_data['Predictions'] = predictions
                result_data['Assessment'] = assessments

                results_placeholder.write("Predictions and Health Assessments")
                results_placeholder.dataframe(result_data)

                # Fetch recovery information if the prediction indicates bad water quality
                if any(pred == 0 for pred in predictions):  # Assuming 0 means bad quality
                    api_key = 'AIzaSyB6bOmdmF6wxUE9xF6mULPh5TYem3yQdic'  # Replace with your actual API key
                    recovery_processor = RecoveryInfoProcessor(api_key)
                    recovery_info = recovery_processor.get_recovery_info("bad water quality")

                    with right_col:
                        right_col.header("Recommended Recovery Steps")
                        if recovery_info:
                            right_col.info(recovery_info)
                        else:
                            right_col.warning("Unable to fetch recovery information.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
