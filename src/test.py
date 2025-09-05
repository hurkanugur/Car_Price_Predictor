import torch
from model import CarPricePredictionModel

def test_model(
    model: "CarPricePredictionModel",
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    norm_params: dict,
    n_samples: int = 10,
):
    """
    Evaluate a trained car price model on the test set and print sample predictions.

    Args:
        model: Trained regression model.
        test_loader: Test set DataLoader.
        device: Computation device ("cpu" or "cuda").
        norm_params: Scalars for de-normalization.
        n_samples: Number of predictions to display (default 10).
    """

    model.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            y_pred = model(X_batch)

            # De-normalize using stored mean/std
            y_pred = y_pred * norm_params["y_std"] + norm_params["y_mean"]
            y_batch = y_batch * norm_params["y_std"] + norm_params["y_mean"]

            all_preds.append(y_pred)
            all_trues.append(y_batch)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)

    # Display sample predictions
    print("\nSample Predicted vs True values:")
    for i in range(min(n_samples, len(all_trues))):
        print(f"Sample {i+1}: Predicted: {all_preds[i].item():.2f}, True: {all_trues[i].item():.2f}")
