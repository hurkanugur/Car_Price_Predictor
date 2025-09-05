import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, random_split
import pickle
import config

class CarPriceDataset:
    """
    Handles loading, splitting, and normalizing car price data for PyTorch.
    """    
    
    def __init__(self, device):
        self.device = device
        self.norm_params = None
        self.categorical_mappings = None

        self.numeric_cols = [
            "Car_Age", 
            "Year", 
            "Engine_Size", 
            "Cylinders",
            "Mileage", 
            "Horsepower", 
            "Doors", 
            "Weight"
        ]

        self.categorical_cols = [
            "Brand",
            "Model",
            "Transmission",
            "Fuel_Type",
            "Color"
        ]

    # ----------------- PUBLIC METHODS -----------------
    def prepare_data_for_training(self):
        """
        Full preprocessing:
        1. Load CSV
        2. Extract features and target
        3. Split train/val/test
        4. Normalize numeric features and target
        5. Create DataLoaders
        Returns:
            train_loader, val_loader, test_loader
        """
        df = self._load_csv()
        X, y = self._extract_features_and_target(df)
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_train_val_test(X, y)
        X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm = self._normalize_train_val_test(X_train, X_val, X_test, y_train, y_val, y_test)
        train_loader, val_loader, test_loader = self._create_dataloaders(X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm)
        return train_loader, val_loader, test_loader
    
    def prepare_data_for_inference(self, df: pd.DataFrame):
        """
        Preprocess a DataFrame for model inference:
        1. Extract features and target
        2. Normalize numeric features and target using stored training stats
        Returns:
            normalized X, y
        """
        X, y = self._extract_features_and_target(df)

        # Retrieve stored normalization parameters
        x_mean, x_std = self.norm_params["x_mean"], self.norm_params["x_std"]
        y_mean, y_std = self.norm_params["y_mean"], self.norm_params["y_std"]

        # Normalize
        X_norm = self._normalize_X(X, x_mean, x_std)
        y_norm = self._normalize_y(y, y_mean, y_std)

        return X_norm, y_norm

    def get_input_dim(self, data_loader):
        """Return number of features dynamically."""
        sample_X, _ = next(iter(data_loader))
        input_dim = sample_X.shape[1]
        print("Input dimension:", input_dim)
        return input_dim

    def save_normalization_params(self):
        """Save normalization parameters."""
        os.makedirs(os.path.dirname(config.NORM_PARAMS_PATH), exist_ok=True)
        with open(config.NORM_PARAMS_PATH, "wb") as f:
            pickle.dump(self.norm_params, f)
        print(f"• Normalization parameters saved to {config.NORM_PARAMS_PATH}")

    def load_normalization_params(self):
        """Load previously saved scalers."""
        with open(config.NORM_PARAMS_PATH, "rb") as f:
            self.norm_params = pickle.load(f)
        print(f"• Normalization parameters loaded from {config.NORM_PARAMS_PATH}")

    def save_categorical_mappings(self):
        """Save categorical mappings separately."""
        os.makedirs(os.path.dirname(config.CATEGORICAL_MAPPINGS_PATH), exist_ok=True)
        with open(config.CATEGORICAL_MAPPINGS_PATH, "wb") as f:
            pickle.dump(self.categorical_mappings, f)
        print(f"• Categorical mappings saved to {config.CATEGORICAL_MAPPINGS_PATH}")

    def load_categorical_mappings(self):
        """Load categorical mappings from file."""
        with open(config.CATEGORICAL_MAPPINGS_PATH, "rb") as f:
            self.categorical_mappings = pickle.load(f)
        print(f"• Categorical mappings loaded from {config.CATEGORICAL_MAPPINGS_PATH}")

    def denormalize_X(self, X_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize numeric features of X using stored training stats.
        Categorical columns remain unchanged.
        """
        x_mean, x_std = self.norm_params["x_mean"], self.norm_params["x_std"]
        n_num_cols = len(self.numeric_cols)
        X_denorm = X_norm.clone()
        X_denorm[:, :n_num_cols] = X_denorm[:, :n_num_cols] * x_std + x_mean
        return X_denorm

    def denormalize_y(self, y_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize target y using stored training stats.
        """
        y_mean, y_std = self.norm_params["y_mean"], self.norm_params["y_std"]
        return y_norm * y_std + y_mean

    # ----------------- PRIVATE METHODS -----------------
    def _load_csv(self):
        """Load CSV dataset into DataFrame."""
        df = pd.read_csv(config.USED_CAR_CSV_PATH)
        print(f"• Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def _extract_features_and_target(self, df: pd.DataFrame):
        """Extract features (X) and target (y) as tensors."""

        df["Car_Age"] = df["Year"].max() - df["Year"]
        df["Price"] = df["Price"].astype(float)

        # Encode categorical features
        if self.categorical_mappings is None:
            # Training: create mappings
            self.categorical_mappings = {}
            for col in self.categorical_cols:
                df[col] = df[col].astype("category")
                self.categorical_mappings[col] = {cat: code for code, cat in enumerate(df[col].cat.categories)}
                df[col] = df[col].cat.codes
        else:
            # Inference: map using stored mappings
            for col in self.categorical_cols:
                df[col] = df[col].map(self.categorical_mappings[col]).fillna(-1).astype(int)

        X = torch.tensor(df[self.numeric_cols + self.categorical_cols].values, dtype=torch.float32)
        y = torch.tensor(df["Price"].values, dtype=torch.float32).reshape(-1, 1)
        return X, y

    def _split_train_val_test(self, X, y):
        """Split dataset into train/val/test sets using random_split."""
        if not config.SPLIT_DATASET:
            return X, X, X, y, y, y

        dataset = TensorDataset(X, y)
        n_total = len(dataset)
        n_train = int(config.TRAIN_SPLIT_RATIO * n_total)
        n_val = int(config.VAL_SPLIT_RATIO * n_total)
        n_test = n_total - n_train - n_val

        generator = torch.Generator().manual_seed(config.SPLIT_RANDOMIZATION_SEED) if config.SPLIT_RANDOMIZATION_SEED is not None else None
        train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

        X_train, y_train = train_ds[:][0], train_ds[:][1]
        X_val, y_val = val_ds[:][0], val_ds[:][1]
        X_test, y_test = test_ds[:][0], test_ds[:][1]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _normalize_train_val_test(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Normalize numeric features and target using pure torch."""

        # stats from training set
        x_mean = X_train[:, :len(self.numeric_cols)].mean(0)
        x_std = X_train[:, :len(self.numeric_cols)].std(0)
        y_mean = y_train.mean()
        y_std = y_train.std()

        # normalize X
        X_train = self._normalize_X(X_train, x_mean, x_std)
        X_val = self._normalize_X(X_val, x_mean, x_std)
        X_test = self._normalize_X(X_test, x_mean, x_std)

        # normalize y
        y_train = self._normalize_y(y_train, y_mean, y_std)
        y_val = self._normalize_y(y_val, y_mean, y_std)
        y_test = self._normalize_y(y_test, y_mean, y_std)

        if self.norm_params is None:
            self.norm_params = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _normalize_X(self, X, x_mean, x_std):
        """Normalize numeric columns of X; leave categorical unchanged."""
        n_num = len(self.numeric_cols)
        X_norm = X.clone()
        X_norm[:, :n_num] = (X_norm[:, :n_num] - x_mean) / (x_std + 1e-8)
        return X_norm

    def _normalize_y(self, y, y_mean, y_std):
        """Normalize target y using mean and std."""
        return (y - y_mean) / (y_std + 1e-8)

    def _create_dataloaders(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Create PyTorch DataLoaders."""
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=config.BATCH_SIZE, shuffle=False)
        return train_loader, val_loader, test_loader
