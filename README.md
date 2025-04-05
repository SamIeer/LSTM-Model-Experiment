<h1>LSTM Traffic Prediction Model Documentation</h1>
<h3>Overview</h3>
<p>This script demonstrates a basic implementation of a Long Short-Term Memory (LSTM) neural network for traffic prediction. It uses synthetic traffic data to train a model that predicts future traffic values (e.g., vehicle counts) based on historical time-series data. The model is built with TensorFlow and Keras, making it a lightweight prototype for smart city traffic forecasting.</p>

<h3>Purpose</h3>
<p>The goal is to create a simple, functional LSTM model as a proof-of-concept for time-series prediction, which can be extended with real traffic datasets  and deployed in a containerized environment like Kubernetes. This serves as a starting point for my GSoC project on traffic prediction microservices.</p>

<h3>Code Structure</h3>
<p>Dependencies<br>
numpy: For numerical operations and data generation.<br>
tensorflow: For building and training the LSTM model.<br>
tensorflow.keras: Provides high-level APIs (Sequential, LSTM, Dense) for neural network construction.</p>
<pre>
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
</pre>

<h3>Data Preparation</h3>
<h4>Synthetic Traffic Data: Randomly generated to simulate hourly vehicle counts.</h>
<p>time_steps = 10: Number of time steps (e.g., 10 hours of historical data per sample).<br>
data: Shape (100, 10, 1) - 100 samples, each with 10 time steps and 1 feature (e.g., vehicle count).<br>
labels: Shape (100, 1) - Target values (e.g., traffic count for the next hour).<br>
Seed: np.random.seed(42) ensures reproducibility.</p>
<pre>
np.random.seed(42)
time_steps = 10
data = np.random.rand(100, time_steps, 1)  # 100 samples, 10 time steps, 1 feature
labels = np.random.rand(100, 1)  # Random target (e.g., next hour's traffic)
</pre>

<h3>Model Architecture</h3>
<h4>Sequential Model: A linear stack of layers.</h4>
<p>LSTM Layer:<br>
50 units (hidden neurons).<br>
activation='relu': Rectified Linear Unit for non-linearity.<br>
input_shape=(time_steps, 1): Matches the data shape (10 time steps, 1 feature).<br>
return_sequences=False: Outputs a single vector per sample (not a sequence).<br>
Dense Layer: 1 unit for a single-value prediction (e.g., next hour’s traffic).</p>
<pre>
  model = Sequential([
    LSTM(50, activation='relu', input_shape=(time_steps, 1), return_sequences=False),
    Dense(1)])
</pre>

<h3>Training</h3>
<h4>Compilation:</h4>
<p>Optimizer: adam (adaptive learning rate).
Loss: mse (mean squared error, suitable for regression tasks like traffic prediction).</p>
<h4>Fitting:</h4>
<p>epochs=5: Trains for 5 iterations over the dataset.
batch_size=32: Processes 32 samples per batch.
verbose=1: Displays training progress.</p>
<pre>
model.compile(optimizer='adam', loss='mse')
model.fit(data, labels, epochs=5, batch_size=32, verbose=1)
</pre>

<h3>Model Saving</h3>
<p>Saves the trained model to a file (traffic_lstm.h5) for later use or deployment.<br>
Prints a confirmation message.</p>
<pre>
model.save('traffic_lstm.h5')
print("Model trained and saved!")
</pre>

<h3>Output:</h3>
<p>Training logs (loss per epoch).<br>
A saved model file.<br>
Message: "Model trained and saved!"</p>
<pre>
  Epoch 1/5
4/4 [==============================] - 1s 10ms/step - loss: 0.3152
Epoch 2/5
4/4 [==============================] - 0s 8ms/step - loss: 0.2854
...
Epoch 5/5
4/4 [==============================] - 0s 7ms/step - loss: 0.2431
Model trained and saved!
</pre>
