from torch import nn
import torch
from sklearn.model_selection import train_test_split

from config import TRAINING_EPOCHS, LEARNING_RATE, VAL_INTERVAL
from data_utils import extract_features_and_target, load_raw_data, normalize_data, split_data
from model_utils import CarPriceRegressor, save_model, save_normalization_params
from plot_utils import LossPlotter

def train_model(model, X_train, y_train, loss_fn, optimizer, X_val, y_val, loss_plotter: LossPlotter):
    """
    Train and monitor the model.
    """
        
    for epoch in range(TRAINING_EPOCHS):
        model.train()                       # Set model to training mode
        optimizer.zero_grad()               # Clear old gradients

        y_pred = model(X_train)             # Forward pass
        loss = loss_fn(y_pred, y_train)     # Compute loss
        loss.backward()                     # Backpropagation
        optimizer.step()                    # Update parameters

        # Record losses
        if (epoch % VAL_INTERVAL == 0) or (epoch == TRAINING_EPOCHS - 1):
            train_loss = loss.item()
            val_loss = validate_model(model, X_val, y_val, loss_fn).item()
            
            # Update live plot
            loss_plotter.update(train_loss=train_loss, val_loss=val_loss)
            print(f"Epoch {epoch} | Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f}")

    return model


def validate_model(model, X_val, y_val, loss_fn):
    """
    Validate the model.
    """

    model.eval()
    with torch.no_grad():
        y_pred = model(X_val)
        loss = loss_fn(y_pred, y_val)

    return loss


def test_model(model, X_test, y_test, norm_params):
    """
    Test the model.
    """

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        
        # De-normalize predicted prices so that we can see in real-life price values
        y_pred = y_pred * norm_params["y_std"] + norm_params["y_mean"]
        y_true = y_test * norm_params["y_std"] + norm_params["y_mean"]

    # Display sample comparison
    print("\nSample Predicted vs True prices:")
    for i in range(10):
        print(f"Predicted: {y_pred[i].item():.2f}, True: {y_true[i].item():.2f}")

def main():
    # Load raw data
    df = load_raw_data()

    # Extract features and target
    X, y = extract_features_and_target(df)
    
    # Split data (training/val/test)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Normalize using TRAINING statistics
    X_train, X_val, X_test, y_train, y_val, y_test, norm_params = normalize_data(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    model = CarPriceRegressor(input_dim=X.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Real-time loss plotter initialization
    loss_plotter = LossPlotter()

    # Train with validation monitoring
    model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        loss_fn=loss_fn,
        optimizer=optimizer,
        X_val=X_val,
        y_val=y_val,
        loss_plotter=loss_plotter
    )

    # Test the model
    test_model(model, X_test, y_test, norm_params)

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

    # Keep the final plot displayed
    loss_plotter.close()

if __name__ == "__main__":
    main()
