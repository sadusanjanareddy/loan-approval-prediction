# Loan Approval Prediction using Machine Learning

## Project Overview
This project builds a Machine Learning model to predict whether a loan application will be **approved or rejected** based on applicant financial and personal information.

The project demonstrates a complete Machine Learning workflow including **data loading, preprocessing, feature scaling, model training, evaluation, and prediction**.

---

## Dataset

The dataset contains information about loan applicants such as:

- Number of dependents
- Education level
- Self employed status
- Annual income
- Loan amount
- Loan term
- CIBIL score
- Residential asset value
- Commercial asset value
- Luxury asset value
- Bank asset value

### Target Variable

`loan_status`

- **1 → Loan Approved**
- **0 → Loan Not Approved**

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  

---

## Machine Learning Model

Algorithm used:

**Logistic Regression**

The model is trained to classify whether a loan application should be approved or rejected based on applicant financial details.

---

## Project Workflow

1. **Data Loading**
   - Load dataset using pandas.

2. **Data Exploration**
   - Check dataset shape
   - Identify column names
   - Detect missing values

3. **Data Preprocessing**
   - Handle missing values using SimpleImputer
   - Encode categorical variables using LabelEncoder
   - Normalize numerical features using MinMaxScaler

4. **Train-Test Split**
   - Split dataset into training and testing sets.

5. **Model Training**
   - Train Logistic Regression model.

6. **Model Evaluation**
   - Accuracy score
   - Confusion matrix
   - Classification report

7. **Prediction**
   - Predict loan approval status for a new applicant.

---

## Model Results

After training the model, the following results were obtained:

```
Target distribution:
 loan_status
0    2656
1    1613

Accuracy: 0.9121779859484778

Confusion Matrix:
[[501  35]
 [ 40 278]]

Classification Report:

              precision    recall  f1-score   support

           0       0.93      0.93      0.93       536
           1       0.89      0.87      0.88       318

    accuracy                           0.91       854
   macro avg       0.91      0.90      0.91       854
weighted avg       0.91      0.91      0.91       854

Loan Status: Approved
```

The model achieved **91% accuracy** on the test dataset.

---

## Project Structure

```
loan-approval-prediction
│
├── model.py
├── data_cleaner.py
├── main.py
├── loan_approval_dataset.csv
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```
git clone https://github.com/your-username/loan-approval-prediction.git
```

Install required libraries

```
pip install -r requirements.txt
```

---

## Run the Project


python main.py


The script will:
- Train the machine learning model
- Evaluate model performance
- Predict loan approval for a sample applicant

---

## Example Prediction
Loan Status: Approved
## Future Improvements

- Implement additional algorithms such as Random Forest and Decision Tree
- Perform hyperparameter tuning
- Add visualization of model results
- Deploy the model as a web application
## Author
**Sadu Sanjana Reddy**

Computer Science Student  
Aspiring Data Scientist
