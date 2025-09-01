import torch
from model_utils import load_model, load_normalization_params
from config import MODEL_PATH, NORM_PARAMS_PATH

def main():
    # Load trained model and normalization parameters
    norm_params = load_normalization_params(NORM_PARAMS_PATH)
    input_dim = len(norm_params["X_mean"])
    model = load_model(input_dim, MODEL_PATH)

    # Age, Mileage, Accident
    X_real = torch.tensor([
        [11, 51000, 1],
        [3, 34742, 1],
        [2, 22372, 0],
        [9, 88900, 0],
    ], dtype=torch.float32)

    # Price
    y_real = torch.tensor([
        [10300],
        [38005],
        [54598],
        [15500]
    ], dtype=torch.float32)

    # Normalize input using training statistics
    X_norm = (X_real - norm_params["X_mean"]) / norm_params["X_std"]

    # Model inference
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(X_norm)
        # De-normalize predicted prices so that we can see in real-life price values
        y_pred = y_pred_norm * norm_params["y_std"] + norm_params["y_mean"]

    # Compare predictions vs actuals
    print("\nPredicted vs Actual Prices:")
    for i in range(X_real.shape[0]):
        print(f"Car {i+1}: Predicted: {y_pred[i].item():.2f}, Actual: {y_real[i].item():.2f}")

if __name__ == "__main__":
    main()
