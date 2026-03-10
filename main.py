import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train,y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    confusion=confusion_matrix(y_test,y_pred)
    report=classification_report(y_test,y_pred)
    metrics = {"accuracy": accuracy,
    "confusion_matrix": confusion,
    "classification_report":report}
    return metrics

def predict_loan_approval(new_applicant, model):
    df_new=pd.DataFrame([new_applicant])
    prediction=model.predict(df_new)[0]
    if prediction==1:
        return "Approved"
    else:
        return  "Not Approved"

# --- Main Execution ---
if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.impute import SimpleImputer

    # TODO: Step 1 – Load your dataset
    df = pd.read_csv('loan_approval_dataset.csv')

    # TODO: Step 2 – Identify columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object','string']).columns.tolist()

    # TODO: Step 3 – Handle missing values
    df[numerical_cols] = SimpleImputer(strategy='mean').fit_transform(df[numerical_cols])
    df[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[categorical_cols])

    # TODO: Step 4 – Encode categorical variables
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # TODO: Step 5 – Normalize numerical columns
    df[numerical_cols] = MinMaxScaler().fit_transform(df[numerical_cols])

    # TODO: Step 6 – Split data
    y = df['loan_status']
    X = df.drop(columns=['loan_id', 'loan_status'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TODO: Step 7 – Train model
    model = train_logistic_regression(X_train, y_train)

    # TODO: Step 8 – Evaluate model
    print("Target distribution:\n", y.value_counts())
    results = evaluate_model(model, X_test, y_test)
    print("Accuracy:", results['accuracy'])
    print("\nConfusion Matrix:\n", results['confusion_matrix'])
    print("\nClassification Report:\n", results['classification_report'])

    # TODO: Step 9 – Predict new applicant
    new_applicant = {
        'no_of_dependents': 2,
        'education': 1,
        'self_employed': 1,
        'income_annum': 40000000,
        'loan_amount': 3000000,
        'loan_term': 15,
        'cibil_score': 700,
        'residential_assets_value': 6000000,
        'commercial_assets_value': 6000000,
        'luxury_assets_value': 75000000,
        'bank_asset_value': 4500000
    }

    result = predict_loan_approval(new_applicant, model)
    print("Loan Status:", result)
