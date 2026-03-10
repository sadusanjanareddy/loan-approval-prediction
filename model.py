import pandas as pd

def load_csv(file_path):
    df=pd.read_csv(r"C:\Users\sanju\Desktop\loan approval\loan_approval_dataset.csv")
    return df

def get_shape(df):
    print("shape of dataset:",df.shape)
    return df.shape

def get_column_names(df):
    print("column names:",df.columns.tolist())
    return df.columns.tolist()

def get_data_types(df):
    print("Data types of all features:")
    return df.dtypes

def get_missing_values(df):
    print("Missing values per column:")
    return df.isnull().sum()

def get_summary(df):
    return df.describe()

# --- Main Execution ---
if __name__ == '__main__':
    file_path = 'loan_approval_dataset.csv'
    df = load_csv(file_path)

    print("Shape of the dataset:")
    print(get_shape(df))

    print("\nColumn names:")
    print(get_column_names(df))

    print("\nData types:")
    print(get_data_types(df))

    print("\nMissing values:")
    print(get_missing_values(df))

    print("\nSummary statistics:")
    print(get_summary(df))

