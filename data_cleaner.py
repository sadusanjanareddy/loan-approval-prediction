import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_csv(file_path):
    df = pd.read_csv(r"C:\Users\sanju\Desktop\loan approval\loan_approval_dataset.csv")

    return df

def get_column_types(df):
    numerical_cols = df.select_dtypes(include=['int64','float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    return numerical_cols, categorical_cols

def impute_missing_values(df, numerical_cols, categorical_cols):
    imputer=SimpleImputer(strategy='mean')
    df[numerical_cols]=imputer.fit_transform(df[numerical_cols])
    imputer=SimpleImputer(strategy='most_frequent')
    df[categorical_cols]=imputer.fit_transform(df[categorical_cols])
    return df

def encode_categorical(df, categorical_cols):
    label_encoder=LabelEncoder()
    for col in categorical_cols:
        df[col]=label_encoder.fit_transform(df[col])
    return df

def scale_numerical(df, numerical_cols):
    scaler=MinMaxScaler()
    df[numerical_cols]=scaler.fit_transform(df[numerical_cols])
    return df

def split_data(df):
    y = df['loan_status']#target
    X = df.drop(['loan_id','loan_status'],axis=1)#features
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train, X_test, y_train, y_test

# --- Main Execution ---
if __name__ == '__main__':
    file_path = 'loan_approval_dataset.csv'

    # Step 1: Load dataset from CSV
    df = load_csv(file_path)

    # Step 2: Identify numerical and categorical columns
    numerical_cols, categorical_cols = get_column_types(df)
    print("Numerical Columns:", numerical_cols)
    print("Categorical Columns:", categorical_cols)
    print()
    
    # Step 3: Handle missing values
    df = impute_missing_values(df, numerical_cols, categorical_cols)

    # Step 4: Encode categorical variables
    print("Encode categorical variables")
    df = encode_categorical(df, categorical_cols)
    print(df.head(5))

    # Step 5: Scale numerical features
    print("Scale numerical features")
    df = scale_numerical(df, numerical_cols)
    print(df.head(5))

    # Step 6: Split dataset into training and testing sets
    print("Split dataset into training and testing sets")
    X_train, X_test, y_train, y_test = split_data(df)

    # Final: Print train and test shapes
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
