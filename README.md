# 🚗 Car Price Prediction with PyTorch

## 📖 Overview
This project predicts **used car prices** using a regression model built in **PyTorch**.  
It covers the complete pipeline from data preprocessing to model inference, including:

- 📊 **Linear Regression Model** implemented with PyTorch  
- ⚖️ **Mean Squared Error (MSE)** as the loss function  
- 🧠 **Adam optimizer** for training  
- 🔀 **Train/Validation/Test split** for robust evaluation  
- 📈 **Feature & target normalization** using training statistics

---

## 🧩 Libraries
- **PyTorch** – model, training, and inference  
- **pandas** – data handling & preprocessing  
- **scikit-learn** – dataset splitting  
- **matplotlib** – plotting loss curves

---

## ⚙️ Requirements

- Python **3.13+**
- Recommended editor: **VS Code**

---

## 📦 Installation

- Clone the repository
```bash
git clone https://github.com/hurkanugur/Car-Price-Predictor.git
```

- Navigate to the `Car-Price-Predictor` directory
```bash
cd Car_Price_Predictor
```

- Install dependencies
```bash
pip install -r requirements.txt
```

- Navigate to the `Car-Price-Predictor/src` directory
```bash
cd src
```

---

## 🔧 Setup Python Environment in VS Code

1. `View → Command Palette → Python: Create Environment`  
2. Choose **Venv** and your **Python version**  
3. Select **requirements.txt** to install dependencies  
4. Click **OK**

---

## 📂 Project Structure

```bash
data/
└── used_cars.csv            # Raw dataset

model/
├── car_price_model.pth      # Trained model (after training)
└── norm_params.pkl          # Normalization params (after training)

src/
├── config.py                # Configurations (paths, hyperparameters, dataset split)
├── data_utils.py            # Data loading, preprocessing, normalization
├── model_utils.py           # Model definition and save/load utilities
├── plot_utils.py            # Loss plotting
├── predict_car_prices.py    # Use the trained model with real-world inputs
├── train_model.py           # Training and evaluation

requirements.txt             # Python dependencies
```
---

## 📂 Train the Model
```bash
python train_model.py
```
or
```bash
python3 train_model.py
```

---

## 📂 Run Predictions on Real Data
```bash
python predict_car_prices.py
```
or
```bash
python3 predict_car_prices.py
```
