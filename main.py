import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

# Define custom CSS for styling
st.markdown("""
    <style>
        /* Background color */
        .reportview-container {
            background-color: #f7f7f9;
        }

        /* Title and text color */
        .title h1 {
            color: #333;
            font-family: 'Helvetica', sans-serif;
        }

        /* Styling sidebar */
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
        }
        
        /* Custom input and button styling */
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border: None;
            padding: 10px 20px;
            border-radius: 10px;
        }
        
        /* Adjust input styles */
        .stTextInput, .stNumberInput, .stSelectbox {
            border: 1px solid #ccc;
            padding: 8px;
            border-radius: 10px;
            font-size: 16px;
        }

        /* Adjust button hover */
        .stButton button:hover {
            background-color: #45a049;
            color: #fff;
        }
    </style>
""", unsafe_allow_html=True)

# Define custom focal loss function
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = tf.keras.backend.epsilon()
        pt_1 = tf.clip_by_value(pt_1, epsilon, 1. - epsilon)
        pt_0 = tf.clip_by_value(pt_0, epsilon, 1. - epsilon)

        return -tf.keras.backend.mean(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1) +
                                      (1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0))
    return focal_loss_fixed

# Load model and preprocessors
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.keras', custom_objects={"focal_loss_fixed": focal_loss(gamma=2.0, alpha=0.25)})

@st.cache_data
def load_encoder():
    return joblib.load('one_hot_encoder.joblib')

@st.cache_data
def load_scaler():
    return joblib.load('scaler.joblib')

model = load_model()
encoder = load_encoder()
scaler = load_scaler()

# Sidebar for input options
st.sidebar.header("Fraud Detection Input")
st.sidebar.write("Please enter transaction details to detect if it's likely fraudulent or not.")

# Transaction data input fields
step = st.sidebar.number_input("Step (Transaction Time Step)", min_value=0)
amount = st.sidebar.number_input("Amount (Transaction Amount)", min_value=0.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance Origin", min_value=0.0)
newbalanceOrig = st.sidebar.number_input("New Balance Origin", min_value=0.0)
oldbalanceDest = st.sidebar.number_input("Old Balance Destination", min_value=0.0)
newbalanceDest = st.sidebar.number_input("New Balance Destination", min_value=0.0)
type_ = st.sidebar.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])

# Display title and welcome text
st.title("🔍 Fraud Detection System")
st.write("""
    Welcome to the Fraud Detection System. This app helps detect potentially fraudulent 
    transactions based on your input data. Please enter the details on the left and 
    click "Predict Fraud" to check if the transaction is fraudulent.
""")

# Prepare the input data
user_data = pd.DataFrame([{
    "step": step,
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest,
    "type": type_
}])

# Apply OneHotEncoder to 'type' column
encoded_type = encoder.transform(user_data[['type']])
encoded_type_columns = encoder.get_feature_names_out(['type'])
encoded_type_df = pd.DataFrame(encoded_type, columns=encoded_type_columns)

# Combine the encoded data with the original input DataFrame (excluding 'type')
user_input_df = pd.concat([user_data.drop(['type'], axis=1), encoded_type_df], axis=1)

# Apply the scaler to numeric columns
numeric_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
user_input_df[numeric_columns] = scaler.transform(user_input_df[numeric_columns])

# Button for prediction
if st.button("Predict Fraud"):
    prediction = model.predict(user_input_df)
    prediction_class = (prediction > 0.5).astype(int)[0][0]  # Convert probability to binary class

    # Display the prediction result with color-coded feedback
    if prediction_class == 1:
        st.error("⚠️ This transaction is likely fraudulent.")
    else:
        st.success("✅ This transaction is likely not fraudulent.")
