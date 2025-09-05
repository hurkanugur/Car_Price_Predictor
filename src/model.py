import torch
from torch import nn
from config import MODEL_PATH

class CarPricePredictionModel(nn.Module):
    """Linear regression model for predicting car prices."""
    def __init__(self, input_dim, device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        # Apply He initialization
        self.net.apply(self.init_weights)

        self.device = device
        self.to(self.device)
        
    def forward(self, x):
        """Forward pass: input features → predicted price."""
        return self.net(x)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
    
    def save(self):
        """Save model state_dict using the path from config."""
        torch.save(self.state_dict(), MODEL_PATH)
        print(f"• Model saved to {MODEL_PATH}")

    def load(self):
        """Load model state_dict using the path from config."""
        self.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.to(self.device)
        self.eval()
        print(f"• Model loaded from {MODEL_PATH}")