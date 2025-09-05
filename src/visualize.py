# visualize.py

import matplotlib.pyplot as plt

class LossMonitor:
    """
    Real-time plotting of training and validation loss.
    """
     
    def __init__(self):
        plt.ion()  # Interactive mode ON
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.train_line, = self.ax.plot([], [], label="Train Loss")
        self.val_line, = self.ax.plot([], [], label="Val Loss")
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("Loss (MSE)")
        self.ax.set_title("Training & Validation Loss")
        self.ax.grid(True)
        self.ax.legend()
        self.train_losses = []
        self.val_losses = []

    def update(self, train_loss, val_loss=None):
        """
        Append new loss values and update the plot.

        Args:
            train_loss (float): training loss for current epoch
            val_loss (float, optional): validation loss for current epoch
        """
        self.train_losses.append(train_loss)
        self.train_line.set_data(range(len(self.train_losses)), self.train_losses)

        if val_loss is not None:
            self.val_losses.append(val_loss)
            self.val_line.set_data(range(len(self.val_losses)), self.val_losses)
        
        # Update axes limits
        self.ax.relim()
        self.ax.autoscale_view()

        plt.draw()
        plt.pause(0.001)  # Small pause to update the figure


    def close(self):
        """
        Keep final plot displayed.
        """
        plt.ioff()
        plt.show()