import torch
import config
from model import CarPricePredictionModel
from visualize import LossMonitor

def train_model(
    model: CarPricePredictionModel,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    loss_monitor: LossMonitor=None,
):
    """
    Train a PyTorch model with mini-batch gradient descent and optional validation.

    Args:
        model (CarPricePredictionModel): Neural network to train.
        train_loader (DataLoader): DataLoader for training dataset.
        val_loader (DataLoader): DataLoader for validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_fn (callable): Loss function (e.g., MSELoss).
        device (torch.device): Device to run training on (CPU or GPU).
        loss_monitor (LossMonitor, optional): Instance to dynamically plot training/validation loss.

    Notes:
        - Validation is performed every `config.VAL_INTERVAL` epochs.
        - Loss values are reported as averages per batch.
    """
    last_val_loss = None  # for plotting

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # -------------------------
        # Training step
        # -------------------------
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)  # average loss per batch

        # -------------------------
        # Validation step
        # -------------------------
        val_loss = None
        if epoch % config.VAL_INTERVAL == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = loss_fn(outputs, y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)  # average validation loss per batch
            last_val_loss = val_loss
            print(f"Epoch {epoch:04d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        else:
            print(f"Epoch {epoch:04d} | Train Loss: {train_loss:.6f}")

        # -------------------------
        # Update LossMonitor
        # -------------------------
        if loss_monitor is not None:
            loss_monitor.update(train_loss, last_val_loss)
