import streamlit as st
import pandas as pd
import joblib
import os
import sys

#add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

try:
    from src.features import apply_feature_engineering
except ImportError:
    st.error("Error: src/features.py not found or could not be imported.")
    st.stop()

#importing the model
model_path = os.path.join("models", "churn_rf_model.pkl")

if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please train the model first.")
    st.stop()
else:  
    rf_model = joblib.load(model_path)

#columns that model expects
if hasattr(rf_model, 'feature_names_in_'):
    expected_columns = rf_model.feature_names_in_
else:
    st.error("Model does not have 'feature_names_in_' attribute. Please ensure the model is trained correctly.")
    st.stop()

#streamlit app
st.set_page_config(page_title="Bank Customer Churn Prediction", page_icon="🏦")
st.title("Bank Customer Churn Prediction")

st.markdown("This application predicts whether a bank customer is likely to churn based on their profile information.")

st.sidebar.header("Input Customer Data")

#function to get user input
def get_user_input():
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 600)
    age = st.sidebar.number_input("Age", 18, 100, 40)
    tenure = st.sidebar.slider("Tenure", 0, 10, 3)
    balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 60000.0)
    salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)
    num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
    geography = st.sidebar.selectbox("Country", ["France", "Germany", "Spain"])
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    card_type = st.sidebar.selectbox("Card Type", ["DIAMOND", "GOLD", "SILVER", "PLATINUM"])
    satisfaction_score = st.sidebar.slider("Satisfaction Score", 1, 5, 3)
    point_earned = st.sidebar.number_input("Point Earned", 0, 10000, 500)
    has_cr_card = st.sidebar.checkbox("Has credit card?", value=True)
    is_active = st.sidebar.checkbox("Is active member?", value=True)

    #create dataframe
    data = {'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': 1 if has_cr_card else 0,
        'IsActiveMember': 1 if is_active else 0,
        'EstimatedSalary': salary,
        'Card Type': card_type,
        'Complain': 0,  #dummy value
        'Satisfaction Score': satisfaction_score,
        'Point Earned': point_earned
    }
    return pd.DataFrame([data])

input_df = get_user_input()

#show user input
st.subheader("Customer Input Data")
st.write(input_df)

#prediction button
if st.button("Predict Churn Probability"):
    #apply feature engineering
    df_processed = apply_feature_engineering(input_df)

    #ensure all expected columns are present
    df_processed = df_processed.reindex(columns=expected_columns, fill_value=0)

    #make prediction
    prediction_proba = rf_model.predict_proba(df_processed)[0][1]

    threshold = 0.40
    is_churn = prediction_proba > threshold

    st.subheader("Prediction Result")

    #display results
    if is_churn:
        st.error(f"The customer is likely to churn with a probability of {prediction_proba*100:.2f}%.")
        st.write("Recommendation: Consider offering retention incentives or personalized services to retain this customer.")

    else:
        st.success(f"The customer is likely to stay with a probability of {(1 - prediction_proba)*100:.2f}%.")
        st.write("Recommendation: Maintain current engagement strategies to keep this customer satisfied.")

    #display processed features for transparency
    with st.expander("Show Processed Features"):
        st.write(df_processed)