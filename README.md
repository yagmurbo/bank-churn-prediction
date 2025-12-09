# Bank Churn Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/yagmurbozatli/bank-churn-prediction)

## Project Overview
This project is an end-to-end Machine Learning application designed to predict whether a bank customer is likely to leave (churn) or stay.

This project was developed for MultiGroup Community's Zero2End ML Course.

The goal is to help banks identify at-risk customers and take proactive measures to retain them. The project includes data preprocessing, feature engineering, model training with **Random Forest**, and a user-friendly web interface built with **Streamlit**.
The main objective was not just to achieve high accuracy but to build a production-ready ML pipeline that solves a real-world business problem.

## App Walkthrough

![ezgif-2bb549005a812c49](https://github.com/user-attachments/assets/3c50de08-31f3-45a4-b339-1cffc63f12ee)


## Technical Methodology (Process Report)

### 1. Baseline & Data Leakage Detection
Initially, the model achieved a suspicious **99% accuracy**. Upon investigation, I identified a **Data Leakage** issue caused by the `Complain` column. Customers who complained almost always churned, which is a post-event indicator, not a predictive feature.
* **Action:** Removed `Complain`, `RowNumber`, `CustomerId`, and `Surname`.
* **Real Baseline:** After cleaning, the raw Random Forest model achieved ~82% Accuracy with low Recall.

### 2. Feature Engineering & Preprocessing
To improve the model's ability to distinguish patterns, I engineered new features based on domain knowledge:
* **`BalanceSalaryRatio`:** (`Balance` / `EstimatedSalary`) - Indicates financial stability.
* **`TenureByAge`:** (`Tenure` / `Age`) - Measures loyalty relative to customer age.
* **`CreditScoreGivenAge`:** (`CreditScore` / `Age`) - Normalizes credit behavior by age.

**Preprocessing Strategy:**
* **Categorical Encoding:** Used One-Hot Encoding for `Geography` and `Card Type` (handling "DIAMOND" etc.), and Label Encoding for `Gender`.
* **Handling Imbalance:** The dataset was imbalanced (Churners were minority). I used `class_weight='balanced'` in the Random Forest model to penalize misclassifying churners.

### 3. Validation Scheme
* **Method:** Train/Test Split (80% Training, 20% Testing).
* **Rationale:** Given the dataset size (10,000 rows), a simple split was sufficient to evaluate performance. `random_state=42` was used to ensure reproducibility.

### 4. Model Performance & Business Alignment
**Final Model vs. Baseline:**
| Metric | Baseline (Leakage Removed) | Final Model (Tuned) |
| :--- | :--- | :--- |
| **Accuracy** | ~82.0% | **86.7%** |
| **Recall (Churn)** | ~45.0% | **~53.0%** |

**Business Alignment:**
In churn prediction, **False Negatives** (missing a customer who leaves) are more costly than False Positives (giving a discount to someone who stays).
* **Optimization:** I adjusted the **Decision Threshold from 0.50 to 0.40**.
* **Result:** This increased the **Recall** significantly, ensuring we catch more risky customers, even if it slightly reduces Precision. This aligns with the business goal of maximizing retention.

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

## Author

**Yağmur Bozatlı**  
[LinkedIn](https://www.linkedin.com/in/yagmur-bozatli/)

