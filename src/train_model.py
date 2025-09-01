from torch import nn
import torch
from sklearn.model_selection import train_test_split

from config import EPOCHS, LR, VAL_INTERVAL
from data_utils import load_raw_data, normalize_data, split_data
from model_utils import CarPriceRegressor, save_model, save_normalization_params
from plot_utils import plot_losses
import matplotlib.pyplot as plt

def train_model(model, X_train, y_train, loss_fn, optimizer, X_val=None, y_val=None):
    """
    Train a PyTorch model on training data and optionally monitor validation loss.

    Args:
        model (nn.Module): PyTorch model to train.
        X_train (Tensor): Training features.
        y_train (Tensor): Training targets.
        loss_fn: Loss function (e.g., nn.MSELoss()).
        optimizer: Optimizer (e.g., torch.optim.Adam).
        X_val (Tensor, optional): Validation features. Defaults to None.
        y_val (Tensor, optional): Validation targets. Defaults to None.

    Returns:
        model (nn.Module): Trained model.
        train_losses (list): Training loss values recorded at each validation interval.
        val_losses (list): Validation loss values recorded at each validation interval.
    """
    
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()

        # Record losses
        if (epoch % VAL_INTERVAL == 0) or (epoch == EPOCHS - 1):
            train_losses.append(loss.item())
            if X_val is not None and y_val is not None:
                model.eval()
                with torch.no_grad():
                    y_val_pred = model(X_val)
                    val_loss = loss_fn(y_val_pred, y_val).item()
                val_losses.append(val_loss)
                print(f"Epoch {epoch} | Train Loss: {loss.item():.10f} | Val Loss: {val_loss:.10f}")
            else:
                val_losses.append(None)
                print(f"Epoch {epoch} | Train Loss: {loss.item():.10f}")

    return model, train_losses, val_losses


def evaluate_model(model, X, y, norm_params=None):
    """
    Compute predictions from a trained model and optionally de-normalize them.

    Args:
        model (nn.Module): Trained PyTorch model.
        X (torch.Tensor): Input features (already normalized if used in training).
        y (torch.Tensor): True target values (already normalized if used in training).
        norm_params (dict, optional): Dictionary with keys "y_mean" and "y_std" for de-normalization. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - y_pred: Model predictions (de-normalized if norm_params provided).
            - y_true: True target values (de-normalized if norm_params provided).
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        y_true = y

        # De-normalize so that we can see the real world values
        if norm_params:
            y_pred = y_pred * norm_params["y_std"] + norm_params["y_mean"]
            y_true = y_true * norm_params["y_std"] + norm_params["y_mean"]

    return y_pred, y_true


def main():
    # Load raw data
    X, y = load_raw_data()
    
    # Split first
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Normalize using TRAINING statistics
    X_train, X_val, X_test, y_train, y_val, y_test, norm_params = normalize_data(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    model = CarPriceRegressor(input_dim=X.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train with validation monitoring
    model, train_losses, val_losses = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        loss_fn=loss_fn,
        optimizer=optimizer,
        X_val=X_val,
        y_val=y_val,
    )

    # Plot train/val loss curves
    plot_losses(train_losses, val_losses)

    # Evaluate on test
    y_pred, y_true = evaluate_model(model, X_test, y_test, norm_params)

    # Display sample comparison
    print("\nSample Predicted vs True prices:")
    for i in range(10):
        print(f"Predicted: {y_pred[i].item():.2f}, True: {y_true[i].item():.2f}")

    # Save model and normalization parameters
    save_model(model)
    save_normalization_params(norm_params)

    # -------------------------
    # Note
    # -------------------------
    print("\n• Note:")
    print("• Dataset: ~4000 samples → predictions might be not precise")
    print("• Model: Linear regression with PyTorch")
    print("• Features: Normalization, train/val/test split")
    print("• Visuals: Training & validation loss curves\n")

if __name__ == "__main__":
    main()
