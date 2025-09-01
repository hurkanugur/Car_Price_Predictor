import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses=None):
    """
    Plot training and validation loss curves.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    if val_losses:
        plt.plot(val_losses, label="Val Loss")        
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()