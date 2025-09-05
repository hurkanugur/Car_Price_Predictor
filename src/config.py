# -------------------------
# Paths
# -------------------------
MODEL_PATH = "../model/car_price_model.pth"
NORM_PARAMS_PATH = "../model/norm_params.pkl"
CATEGORICAL_MAPPINGS_PATH = "../model/categorical_mappings.pkl"
USED_CAR_CSV_PATH = "../data/car_price_dataset.csv"

# -------------------------
# Training hyperparameters
# -------------------------
LEARNING_RATE = 5e-4       # Learning rate
WEIGHT_DECAY = 5e-4       # Weight decay (L2 regularization) for Adam
BATCH_SIZE = 64           # Batch size
NUM_EPOCHS = 1000          # Number of training epochs
VAL_INTERVAL = 1           # Interval for validation loss evaluation

# -------------------------
# Data split ratios
# -------------------------
SPLIT_DATASET = True                # Set to False to use the same dataset for train/val/test
SPLIT_RANDOMIZATION_SEED = 42       # Seed for reproducible shuffling; set to None for random behavior
TRAIN_SPLIT_RATIO = 0.7             # Train dataset split ratio
VAL_SPLIT_RATIO = 0.15              # Validation dataset split ratio
TEST_SPLIT_RATIO = 0.15             # Test dataset split ratio