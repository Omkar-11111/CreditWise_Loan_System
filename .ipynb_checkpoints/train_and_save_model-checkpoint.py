"""
Trains the loan approval model using the exact same preprocessing steps
as credit_wise.ipynb, and saves everything the Streamlit app needs to
make predictions on new applicant data:
    - model.pkl        -> trained Logistic Regression model
    - scaler.pkl        -> StandardScaler fitted on training data
    - imputer_values.pkl -> mean/mode values used to fill missing data
    - encoders.pkl       -> LabelEncoder + OneHotEncoder + final column order
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# 1. Load data
df = pd.read_csv("loan_approval_data.csv")

# 2. Handle missing values (same as notebook)
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

# Save the imputer statistics so the app can fill in any blank fields the same way
imputer_values = {
    "numerical": dict(zip(numerical_cols, num_imp.statistics_)),
    "categorical": dict(zip(categorical_cols, cat_imp.statistics_)),
}

# 3. Drop ID column
df = df.drop("Applicant_ID", axis=1)

# 4. Encoding (same as notebook, but using separate encoders per column
#    so we can correctly recover each column's classes later)
le_edu = LabelEncoder()
df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])

le_target = LabelEncoder()
df["Loan_Approved"] = le_target.fit_transform(df["Loan_Approved"])  # No=0, Yes=1

ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(df[ohe_cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)
df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

# 5. Train/test split
X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]
final_columns = X.columns.tolist()  # exact column order the model expects

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train final model (Logistic Regression - best performer)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print("Final Model: Logistic Regression")
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8. Save everything the app needs
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer_values, "imputer_values.pkl")
joblib.dump(
    {
        "label_encoder_education": le_edu,
        "education_classes": list(le_edu.classes_),
        "target_classes": list(le_target.classes_),  # e.g. ['No', 'Yes'] -> 0, 1
        "ohe": ohe,
        "ohe_cols": ohe_cols,
        "final_columns": final_columns,
    },
    "encoders.pkl",
)

print("\nSaved: model.pkl, scaler.pkl, imputer_values.pkl, encoders.pkl")
