import torch
from torch import nn
import os
import pickle

from config import MODEL_PATH, NORM_PARAMS_PATH

class CarPriceRegressor(nn.Module):
    """Linear regression model for predicting car prices."""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        """Forward pass: input features → predicted price."""
        return self.linear(x)
    
def save_model(model, path=MODEL_PATH):
    """Save model parameters to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"\n• Model has been saved: {MODEL_PATH}")


def load_model(input_dim, path=MODEL_PATH):
    """Load model parameters from a file."""
    model = CarPriceRegressor(input_dim)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def save_normalization_params(norm_params, path=NORM_PARAMS_PATH):
    """Save normalization stats (mean, std) to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(norm_params, f)
    print(f"• Normalization parameters have been saved: {NORM_PARAMS_PATH}")

def load_normalization_params(path=NORM_PARAMS_PATH):
    """Load normalization stats from a file."""
    with open(path, "rb") as f:
        return pickle.load(f)
