# Bank Churn Prediction

## Project Overview
This project is an end-to-end Machine Learning application designed to predict whether a bank customer is likely to leave (churn) or stay.

This project was developed for MultiGroup Community's Zero2End ML Course.

The goal is to help banks identify at-risk customers and take proactive measures to retain them. The project includes data preprocessing, feature engineering, model training with **Random Forest**, and a user-friendly web interface built with **Streamlit**.
The main objective was not just to achieve high accuracy but to build a production-ready ML pipeline that solves a real-world business problem.

## Dataset
This dataset is Bank Customer Churn dataset from Kaggle. https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn

## App Walkthrough

![ezgif-2bb549005a812c49](https://github.com/user-attachments/assets/3c50de08-31f3-45a4-b339-1cffc63f12ee)


## Technical Methodology

### 1. Baseline & Data Leakage Detection
Training with the original data, model achieved a **99%** accuracy. By looking at feature importances, I discovered that this is because of `Complain` column. Customers who complained always churned, so this is not a predictive feature.  
Removed `RowNumber`, `CustomerId`, and `Surname` columns since they are irrelevant.

### 2. Feature Engineering & Preprocessing
To improve the model's ability to catch patterns, I added new features:  
* **`BalanceSalaryRatio`:** (`Balance` / `EstimatedSalary`) - Indicates financial stability.
* **`TenureByAge`:** (`Tenure` / `Age`) - Measures loyalty relative to age.
* **`CreditScoreGivenAge`:** (`CreditScore` / `Age`) - Normalizes credit behavior by age.
* **Categorical Encoding:** Used One-Hot Encoding for `Geography`, `Card Type`, and `Gender`.

* **Interactive UI:** Deployed on **Hugging Face Spaces** using Streamlit for easy access.

## Tech Stack
* **Language:** Python 3.10
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (Random Forest Classifier)
* **Web Framework:** Streamlit
* **Version Control:** Git

## Model Performance
After fixing data leakage and applying feature engineering:
* **Accuracy:** ~86.7%

## Project Structure
```text
bank-churn-prediction/

├── notebooks/             
│   └── baseline.ipynb     # model training
│   └── EDA.ipynb          # explatory data analysis
│   └── rf_model.pkl       # final model
├── src/                   
│   └── features.py        # preprocessing and feature engineering
├── app.py                 # Streamlit web application
├── requirements.txt       # project dependencies
└── README.md              # project documentation

```

## How to Run Locally

**Clone the repository:**

```text

git clone https://github.com/yagmurbo/bank-churn-prediction.git

```

**Install dependencies:**

```text
pip install -r requirements.txt

```

**Run the Streamlit app:**

```text
streamlit run app.py
```

