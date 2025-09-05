import pandas as pd
import torch
from dataset import CarPriceDataset
from model import CarPricePredictionModel

def main():
    # -------------------------
    # Select device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")

    # -------------------------
    # Load dataset normalization parameters and categorical mappings
    # -------------------------
    dataset = CarPriceDataset(device=device)
    dataset.load_normalization_params()
    dataset.load_categorical_mappings()

    # -------------------------
    # Load trained model
    # -------------------------
    model = CarPricePredictionModel(input_dim=13, device=device)
    model.load()

    # -------------------------
    # Example real-world input
    # -------------------------
    df = pd.DataFrame([
        {
            "Brand": "Honda", "Model": "Civic", "Year": 2004, "Engine_Size": 5.0, "Cylinders": 8,
            "Transmission": "Automatic", "Fuel_Type": "Electric", "Mileage": 56750,
            "Horsepower": 264, "Doors": 4, "Weight": 2282, "Color": "Silver", "Price": 56750
        },
        {
            "Brand": "Ford", "Model": "Fusion", "Year": 1991, "Engine_Size": 5.0, "Cylinders": 10,
            "Transmission": "Manual", "Fuel_Type": "Diesel", "Mileage": 45938,
            "Horsepower": 326, "Doors": 2, "Weight": 1259, "Color": "Blue", "Price": 45938
        },
        {
            "Brand": "Honda", "Model": "Civic", "Year": 2016, "Engine_Size": 4.4, "Cylinders": 4,
            "Transmission": "Manual", "Fuel_Type": "Hybrid", "Mileage": 48707,
            "Horsepower": 239, "Doors": 3, "Weight": 1362, "Color": "White", "Price": 48707
        },
        {
            "Brand": "Toyota", "Model": "Prius", "Year": 1996, "Engine_Size": 1.9, "Cylinders": 12,
            "Transmission": "Manual", "Fuel_Type": "Hybrid", "Mileage": 14659,
            "Horsepower": 63, "Doors": 3, "Weight": 2841, "Color": "Red", "Price": 14659
        }
    ])

    # -------------------------
    # Prepare data for inference
    # -------------------------
    X_norm, _ = dataset.prepare_data_for_inference(df=df)
    X_norm = X_norm.to(device)

    # -------------------------
    # Model inference
    # -------------------------
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(X_norm)
        # Denormalize predictions
        y_pred = dataset.denormalize_y(y_pred_norm)

    # -------------------------
    # Display predictions
    # -------------------------
    print("\nPredicted Prices:")
    for i in range(len(y_pred)):
        print(f"Car {i+1}: Predicted Price: {y_pred[i].item():.2f}")

if __name__ == "__main__":
    main()
