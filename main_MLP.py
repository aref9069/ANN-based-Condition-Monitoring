import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Clear any previous sessions
keras.backend.clear_session()

# Load data from .mat files
kurtosis_data = loadmat('KURTOSISfit.mat')
rms_data = loadmat('RMSfit.mat')
time_data = loadmat('timeT5.mat')

# Extract arrays (adjust key names based on your .mat file structure)
KURfit = kurtosis_data['KURfit']
RMSfit = rms_data['RMSfit']
time = time_data['time']

# Combine inputs
inputs = np.vstack([KURfit, RMSfit])
targets = time

# Transpose if needed to get (samples, features) shape
if inputs.shape[0] == 2:
    inputs = inputs.T
if targets.ndim == 2 and targets.shape[0] == 1:
    targets = targets.T

# Split data: 75% train, 15% validation, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    inputs, targets, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, random_state=42
)

# Scale the data (equivalent to mapminmax in MATLAB)
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()

X_train_scaled = input_scaler.fit_transform(X_train)
X_val_scaled = input_scaler.transform(X_val)
X_test_scaled = input_scaler.transform(X_test)

y_train_scaled = output_scaler.fit_transform(y_train.reshape(-1, 1))
y_val_scaled = output_scaler.transform(y_val.reshape(-1, 1))
y_test_scaled = output_scaler.transform(y_test.reshape(-1, 1))

# Build the neural network
# Hidden layer with 7 neurons and tanh activation, output layer with linear activation
model = keras.Sequential([
    layers.Dense(7, activation='tanh', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(1, activation='linear')
])

# Compile the model (using MSE as performance function, similar to trainlm)
model.compile(
    optimizer='adam',  # Adam is a good alternative to Levenberg-Marquardt
    loss='mse',
    metrics=['mae']
)

# Early stopping (similar to max_fail=20)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# Train the network
print("Training the network...")
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Make predictions
train_pred_scaled = model.predict(X_train_scaled)
val_pred_scaled = model.predict(X_val_scaled)
test_pred_scaled = model.predict(X_test_scaled)

# Inverse transform predictions to original scale
train_pred = output_scaler.inverse_transform(train_pred_scaled)
val_pred = output_scaler.inverse_transform(val_pred_scaled)
test_pred = output_scaler.inverse_transform(test_pred_scaled)

# Calculate performance metrics
train_mse = np.mean((y_train - train_pred) ** 2)
val_mse = np.mean((y_val - val_pred) ** 2)
test_mse = np.mean((y_test - test_pred) ** 2)

print(f"\nPerformance (MSE):")
print(f"Training: {train_mse:.6f}")
print(f"Validation: {val_mse:.6f}")
print(f"Test: {test_mse:.6f}")

# Plot training history (similar to plotperform)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training Performance')
plt.legend()
plt.grid(True)
plt.show()

# Plot regression results (similar to plotregression)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Training data
axes[0, 0].scatter(y_train, train_pred, alpha=0.5)
axes[0, 0].plot([y_train.min(), y_train.max()], 
                [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Target')
axes[0, 0].set_ylabel('Output')
axes[0, 0].set_title(f'Train Data (R² = {np.corrcoef(y_train.flatten(), train_pred.flatten())[0,1]**2:.4f})')
axes[0, 0].grid(True)

# Validation data
axes[0, 1].scatter(y_val, val_pred, alpha=0.5)
axes[0, 1].plot([y_val.min(), y_val.max()], 
                [y_val.min(), y_val.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Target')
axes[0, 1].set_ylabel('Output')
axes[0, 1].set_title(f'Validation Data (R² = {np.corrcoef(y_val.flatten(), val_pred.flatten())[0,1]**2:.4f})')
axes[0, 1].grid(True)

# Test data
axes[1, 0].scatter(y_test, test_pred, alpha=0.5)
axes[1, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Target')
axes[1, 0].set_ylabel('Output')
axes[1, 0].set_title(f'Test Data (R² = {np.corrcoef(y_test.flatten(), test_pred.flatten())[0,1]**2:.4f})')
axes[1, 0].grid(True)

# All data
all_targets = np.vstack([y_train, y_val, y_test])
all_predictions = np.vstack([train_pred, val_pred, test_pred])
axes[1, 1].scatter(all_targets, all_predictions, alpha=0.5)
axes[1, 1].plot([all_targets.min(), all_targets.max()], 
                [all_targets.min(), all_targets.max()], 'r--', lw=2)
axes[1, 1].set_xlabel('Target')
axes[1, 1].set_ylabel('Output')
axes[1, 1].set_title(f'All Data (R² = {np.corrcoef(all_targets.flatten(), all_predictions.flatten())[0,1]**2:.4f})')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Save the model
model.save('trained_model.h5')
print("\nModel saved as 'trained_model.h5'")