import pandas as pd
import sys
import os
import numpy as np
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.features import apply_feature_engineering
except ImportError as e:
    print(f"Error importing apply_feature_engineering: {e}")
    sys.exit(1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    file_path = os.path.join("data", "Customer-Churn-Records.csv")

    if not os.path.exists(file_path):
        print(f"Data file not found at {file_path}")
        return
    
    print("Loading data...")
    df = pd.read_csv(file_path)

    target_col = "Exited"
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in data")
        return
    
    print("Applying feature engineering...")
    df_processed = apply_feature_engineering(df)

    print("Splitting data into train and test sets...")
    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Classifier (Balanced)...")
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    print("Evaluating model with custom threshold...")
    y_probs = rf_model.predict_proba(X_test)[:, 1]

    #custom thresholding: if prob > 0.4, predict 1 else 0
    threshold = 0.4
    y_pred_new = (y_probs > threshold).astype(int)

    acc = accuracy_score(y_test, y_pred_new)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_new))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_new)
    print(cm)
    print(f"Churn Number: {cm[1,1]}")

    #feature importance
    feature_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nMost important 5 features:")
    print(feature_imp.head(5))

    #live prediction simulation
    print("New customer prediction simulation:")

    new_customer = {
        'CreditScore': 600,
        'Geography': 'Germany',
        'Gender': 'Male',
        'Age': 40,
        'Tenure': 3,
        'Balance': 60000,
        'NumOfProducts': 2,
        'HasCrCard': 1,
        'IsActiveMember': 1,
        'EstimatedSalary': 50000,
        'Card Type': 'DIAMOND'
    }

    df_new = pd.DataFrame([new_customer])
    print(f"New customer data: {df_new.iloc[0].to_dict()}")
    df_new_processed = apply_feature_engineering(df_new)
    df_new_processed = df_new_processed.reindex(columns=X.columns, fill_value=0)
    prob= rf_model.predict_proba(df_new_processed)[0][1]
    pred = 1 if prob > threshold else 0

    print("Prediction for new customer:")
    print(f"Churn Probability: {prob*100:.2f}%")
    if pred == 1:
        print("Predicted to Churn.")
    else:
        print("Predicted to Stay.")

    print("Saving model...")
    if not os.path.exists("models"):
        os.makedirs("models")

    #save model as .pkl file
    joblib.dump(rf_model, "models/churn_rf_model.pkl")

    print("Model successfully saved.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")

    print("Script finished execution.")

