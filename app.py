import pandas as pd
import numpy as np
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# Streamlit Page Configuration
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")
st.title("Breast Cancer Prediction App")
st.markdown("Predict whether a tumor is **malignant** or **benign** based on input features.")

# Load the dataset
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# Create a DataFrame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['Diagnosis'] = breast_cancer_dataset.target

# Show basic details about the data
st.subheader("Dataset Overview")
with st.expander("Click to view dataset details"):
    st.write("First 5 rows of the dataset:")
    st.dataframe(data_frame.head())

# Split data into features and target
X = data_frame.drop(columns='Diagnosis', axis=1)
Y = data_frame['Diagnosis']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model Training
model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)

# Evaluate Accuracy
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)

# Display model performance
st.subheader("Model Performance")
st.write(f"Accuracy on Training Data: **{training_data_accuracy:.2f}**")
st.write(f"Accuracy on Testing Data: **{testing_data_accuracy:.2f}**")

# User Input for Prediction
st.subheader("Make a Prediction")
st.markdown("Enter the required tumor features below to predict if the tumor is malignant or benign.")

# Create a grid layout for input
with st.form("prediction_form"):
    st.write("### Tumor Features")
    user_input = []
    cols = st.columns(5)  # Adjust the number of columns to fit the layout

    # Loop through features and assign to grid columns
    for idx, feature in enumerate(breast_cancer_dataset.feature_names):
        col = cols[idx % 5]  # Distribute fields across columns
        value = col.number_input(f"{feature}", value=0.0, step=0.1, format="%.4f")
        user_input.append(value)
    
    # Form submit button
    submitted = st.form_submit_button("Predict")

# Process Prediction
if submitted:
    try:
        # Convert user input into a NumPy array
        input_data_as_numpy_array = np.array(user_input).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data_as_numpy_array)
        
        # Display result
        if prediction[0] == 1:
            st.success("The tumor is **Malignant**.")
        else:
            st.success("The tumor is **Benign**.")
    except Exception as e:
        st.error("There was an error processing your input. Please try again.")
