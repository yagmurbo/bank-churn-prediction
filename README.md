# Bank Churn Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/yagmurbozatli/bank-churn-prediction)

## Project Overview
This project is an end-to-end Machine Learning application designed to predict whether a bank customer is likely to leave (churn) or stay.

This project was developed for MultiGroup Community's Zero2End ML Course.

The goal is to help banks identify at-risk customers and take proactive measures to retain them. The project includes data preprocessing, feature engineering, model training with **Random Forest**, and a user-friendly web interface built with **Streamlit**.

🔗 **Live Demo:** [Click here to test the app!](https://huggingface.co/spaces/yagmurbozatli/bank-churn-prediction)

## Key Features
* **Data Leakage Prevention:** Identified and removed the `Complain` column, which was causing data leakage and unrealistic accuracy scores (99%).
* **Advanced Feature Engineering:** Created new features to capture customer behavior better:
    * `BalanceSalaryRatio`: Financial stability indicator.
    * `TenureByAge`: Loyalty metric.
    * `CreditScoreGivenAge`: Credit behavior relative to age.
* **Class Imbalance Handling:** Used `class_weight='balanced'` and optimized the **Decision Threshold** (lowered to 0.40) to improve the Recall rate for identifying churners.
* **Model Serialization:** The trained model is saved using `joblib` to ensure fast inference without retraining.
* **Interactive UI:** Deployed on **Hugging Face Spaces** using Streamlit for easy access.

## Tech Stack
* **Language:** Python 3.10+
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (Random Forest Classifier)
* **Web Framework:** Streamlit
* **Deployment:** Hugging Face Spaces & Docker
* **Version Control:** Git & GitHub

## Model Performance
After fixing data leakage and applying feature engineering:
* **Accuracy:** ~86.7%
* **Recall (Churn Class):** ~53% (Improved via threshold tuning)
* **Precision:** Balanced to minimize false alarms while catching potential churners.

## Project Structure
```text
bank-churn-prediction/
├── data/                  # Dataset files
├── models/                # Saved models (.pkl files)
├── src/                   # Source code for feature engineering
│   └── features.py        # Preprocessing & Feature Engineering logic
├── app.py                 # Streamlit web application
├── main.py                # Model training and evaluation script
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

```

## How to Run Locally

**Clone the repository:**

git clone [https://github.com/yagmurbo/bank-churn-prediction.git](https://github.com/yagmurbo/bank-churn-prediction.git)
cd bank-churn-prediction

**Install dependencies:**

pip install -r requirements.txt

**Run the Streamlit app:**

streamlit run app.py