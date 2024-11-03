import os
import torch
import numpy as np
from kan import KAN
from get_data import get_data
from save_metrics import save_metrics

# GPU or CPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

# Model and data parameters
ticker = "AMZN"
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
start_date = '2017-01-01'
end_date = '2023-01-01'
seq_train_length = 60
steps_to_predict = 1
scale = True
test_size = 0.2
save_data = False
split_by_date = False
model_path = "kan_model.pth"  # Model save path
num_days_to_predict = 4  # Set the number of days you want to predict

# Step 1: Load and Prepare Data
d_r = get_data(
    ticker,
    feature_columns,
    start_date,
    end_date,
    seq_train_length,
    num_days_to_predict,
    scale,
    test_size,
    save_data,
    split_by_date
)

data_df, result_df = d_r[0], d_r[1]

x_train = result_df['X_train']
y_train = result_df['y_train']
x_test = result_df['X_test']
y_test = result_df['y_test']
test_dates = result_df["test_dates"]  # Extract test dates
column_scaler = result_df['column_scaler']

# Limit the predictions to the last few days in the test set
x_test_subset = x_test[-num_days_to_predict:]
y_test_subset = y_test[-num_days_to_predict:]
test_dates_subset = test_dates[-num_days_to_predict:]

# Reshape data for model input
x_train = x_train.reshape(x_train.shape[0], -1)
x_test_subset = x_test_subset.reshape(x_test_subset.shape[0], -1)

# Prepare dataset tensors for the subset
dataset = {
    'train_input': torch.tensor(x_train).float().to(device),
    'test_input': torch.tensor(x_test_subset).float().to(device),
    'train_label': torch.tensor(y_train).float().to(device).unsqueeze(1),
    'test_label': torch.tensor(y_test_subset).float().to(device).unsqueeze(1)
}

input_size = dataset['train_input'].shape[1]
num_neurons = len(dataset['train_input']) // 50
print(len(dataset['train_input']))
output_size = 1
#
# Model initialization
model = KAN(width=[input_size, num_neurons*3, output_size], grid=3, k=3, seed=0, device=device)

# Step 2: Load or Train Model
# if os.path.exists(model_path):
#     print(f"Loading existing model from {model_path}")
#     model.load_state_dict(torch.load(model_path, map_location=device))
# else:
print("Training the model...")
summary = model.fit(dataset, opt="LBFGS", steps=10)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Step 3: Predict and Inverse Transform Predictions
with torch.no_grad():
    predictions = model(dataset['test_input']).cpu().numpy()

# Apply inverse transform to convert back to actual price values
predicted_prices = column_scaler['Close'].inverse_transform(predictions)
actual_prices = column_scaler['Close'].inverse_transform(y_test_subset.reshape(-1, 1))
predicted_prices = np.maximum(predicted_prices, 0)  # Clamp negative values

# Step 4: Print Predicted and Actual Prices with Dates for the Selected Days
print("\nPredicted Prices vs Actual Prices (Last Few Days):")
print("{:<15} {:<20} {:<20} {:<20}".format("Date", "Index", "Predicted Price", "Actual Price"))
print("="*75)
for i, date in enumerate(test_dates_subset):
    print(f"{date:<15} {i:<20} {predicted_prices[i][0]:<20.2f} {actual_prices[i][0]:<20.2f}")

save_metrics(model, dataset['train_input'], dataset['train_label'], dataset['test_input'], dataset['test_label'], "KAN_Model")