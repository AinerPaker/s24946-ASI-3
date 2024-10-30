import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression


def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print('Cleaned data loaded successfully.')
        return data
    except Exception as e:
        print(f'Error loading cleaned data: {e}')
        return None


def encode_categorical_columns(data, categorical_columns):
    label_encoders = {}
    for col in categorical_columns:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
    return data, label_encoders


def split_data(data, target_column, test_size=0.2, random_state=42):
    if target_column not in data.columns:
        print(f"'{target_column}' column not found in data.")
        return None, None, None, None

    X = data.drop([target_column, 'rownames'], axis=1, errors='ignore')
    y = data[target_column]

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    except ValueError as e:
        print(f'Error splitting data: {e}')
        return None, None, None, None


def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return r2, mae, mse


def main():
    cleaned_data_path = 'cleaned_data.csv'
    categorical_columns = ['gender', 'ethnicity', 'fcollege', 'mcollege', 'home', 'urban', 'region', 'education',
                           'income']
    target_column = 'score'

    data = load_data(cleaned_data_path)
    if data is None:
        return

    data, label_encoders = encode_categorical_columns(data, categorical_columns)
    X_train, X_test, y_train, y_test = split_data(data, target_column)

    if X_train is None or X_test is None or y_train is None or y_test is None:
        return

    r2, mae, mse = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    print("Model Evaluation Metrics:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")


if __name__ == '__main__':
    main()
