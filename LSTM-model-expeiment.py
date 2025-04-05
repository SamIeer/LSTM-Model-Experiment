import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Synthetic traffic data (e.g., hourly vehicle counts)
np.random.seed(42)
time_steps = 10
data = np.random.rand(100, time_steps, 1)  # 100 samples, 10 time steps, 1 feature
labels = np.random.rand(100, 1)  # Random target (e.g., next hour's traffic)

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(time_steps, 1), return_sequences=False),
    Dense(1)
])

# Compile and train
model.compile(optimizer='adam', loss='mse')
model.fit(data, labels, epochs=5, batch_size=32, verbose=1)

# Save model
model.save('traffic_lstm.h5')
print("Model trained and saved!")