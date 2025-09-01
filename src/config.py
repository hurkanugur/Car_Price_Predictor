# -------------------------
# Paths
# -------------------------
MODEL_PATH = "../model/car_price_model.pth"
NORM_PARAMS_PATH = "../model/norm_params.pkl"
USED_CAR_CSV_PATH = "../data/used_cars.csv"

# -------------------------
# Training hyperparameters
# -------------------------
LR = 0.0001           # Learning rate
EPOCHS = 50000       # Number of training epochs
VAL_INTERVAL = 1000   # Interval for validation loss evaluation

# -------------------------
# Data split ratios
# -------------------------
SPLIT_DATASET = True  # Set to False to use the same dataset for train/val/test
SPLIT_RANDOM_STATE = 42 # Seed for reproducible shuffling; set to None for random behavior
TRAIN_SET_SIZE = 0.7
VAL_SET_SIZE = 0.15
TEST_SET_SIZE = 0.15