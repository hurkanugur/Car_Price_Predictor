import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from config import SPLIT_DATASET, SPLIT_RANDOM_STATE, TEST_SET_SIZE, TRAIN_SET_SIZE, VAL_SET_SIZE, USED_CAR_CSV_PATH

def load_raw_data():
    """
    Load CSV data and extract features and target without normalization.
    
    Returns:
        X (Tensor): Raw features
        y (Tensor): Raw target
    """

    df = pd.read_csv(USED_CAR_CSV_PATH)
    print(f"\n• Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")
    print(f"{df.head()}\n")

    # Feature 1: Car age
    age = df["model_year"].max() - df["model_year"]

    # Feature 2: Mileage
    mileage = df["milage"].str.replace(",", "").str.replace(" mi.", "").astype(int)

    # Feature 3: Accident
    accident = (df["accident"] != "None reported").astype(int)

    # Target: Price
    price = df["price"].str.replace("$", "").str.replace(",", "").astype(int)

    X = torch.column_stack([
        torch.tensor(age, dtype=torch.float32),
        torch.tensor(mileage, dtype=torch.float32),
        torch.tensor(accident, dtype=torch.float32)
    ])

    y = torch.tensor(price, dtype=torch.float32).reshape((-1, 1))

    return X, y

def split_data(X, y):
    """
    Split raw data into train/val/test sets.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """

    if not SPLIT_DATASET:
        print("• Dataset splitting disabled. Using the same dataset for train/val/test.\n")
        return X, X, X, y, y, y
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - TRAIN_SET_SIZE), random_state=SPLIT_RANDOM_STATE, shuffle=True
    )
    relative_test_size = TEST_SET_SIZE / (VAL_SET_SIZE + TEST_SET_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=SPLIT_RANDOM_STATE, shuffle=True
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def normalize_data(X_train, X_val=None, X_test=None, y_train=None, y_val=None, y_test=None):
    """
    Normalize features and target using TRAINING set statistics.

    Returns:
        X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm, norm_params
    """

    # Features
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std if X_val is not None else None
    X_test_norm = (X_test - X_mean) / X_std if X_test is not None else None

    # Target
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std if y_val is not None else None
    y_test_norm = (y_test - y_mean) / y_std if y_test is not None else None

    norm_params = {
        "X_mean": X_mean, 
        "X_std": X_std, 
        "y_mean": y_mean, 
        "y_std": y_std
    }

    return X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm, norm_params
