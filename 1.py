import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Get all CSV filenames in the directory
path = 'training_data_new'
file_names = [f for f in os.listdir(path) if f.endswith('.csv')]
file_names.sort(key=lambda f: int(f.split('.')[0]))  # Sort filenames based on the numeric part before '.csv'

# Load the dataset
all_data = []
for file_name in file_names:
    df = pd.read_csv(os.path.join(path, file_name))
    all_data.append(df)

data = pd.concat(all_data)

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Split data into sequences of 67 time steps
X, Y = [], []
for i in range(len(data_scaled) - 67 - 5):  # 5 for the prediction horizon
    X.append(data_scaled[i:i+67, :])
    Y.append(data_scaled[i+67:i+67+5, :2])  # Only the (x,y) coordinates

X = np.array(X)
Y = np.array(Y)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, Reshape

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(67, 12)),
    LSTM(128),
    Dense(5*2, activation='linear'),  # 5 time steps * 2 coordinates
    Reshape((5, 2))  # Reshaping the output to match the target shape
])
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, Y_train, epochs=50, batch_size=64, validation_data=(X_test, Y_test))

# Predictions
predictions = model.predict(X_test)

# Inverse transform to get original coordinate values
# Construct a zero array for the remaining columns (except x and y) to inverse transform
zero_cols = np.zeros((Y_test.shape[0] * Y_test.shape[1], data_scaled.shape[1] - 2))
combined = np.hstack([predictions.reshape(-1, 2), zero_cols])
predictions_rescaled = scaler.inverse_transform(combined)[:, :2].reshape(Y_test.shape)

# Do the same for Y_test
combined_Y = np.hstack([Y_test.reshape(-1, 2), zero_cols])
Y_test_rescaled = scaler.inverse_transform(combined_Y)[:, :2].reshape(Y_test.shape)

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions_rescaled - Y_test_rescaled)**2))
print(f"RMSE: {rmse}")
