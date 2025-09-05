from torch import nn
import torch

from config import LEARNING_RATE, WEIGHT_DECAY
from dataset import CarPriceDataset
from model import CarPricePredictionModel
import train
import test
from visualize import LossMonitor

def main():

    # Select CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    
    # Load and prepare data
    car_price_dataset = CarPriceDataset(device=device)
    train_loader, val_loader, test_loader = car_price_dataset.prepare_data_for_training()

    # Initialize model, optimizer, loss
    input_dim = car_price_dataset.get_input_dim(train_loader)
    model = CarPricePredictionModel(input_dim=input_dim, device=device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Initialize LossMonitor
    loss_monitor = LossMonitor()

    # Train the model 
    train.train_model(model, train_loader, val_loader, optimizer, loss_fn, device, loss_monitor)

    # Test the model
    test.test_model(model, test_loader, device, car_price_dataset.norm_params)

    # Save model normalization parameters and categorical mappings
    model.save()
    car_price_dataset.save_normalization_params()
    car_price_dataset.save_categorical_mappings()

    # Keep the final plot displayed
    loss_monitor.close()

if __name__ == "__main__":
    main()
