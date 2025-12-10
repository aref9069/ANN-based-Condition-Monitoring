import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Clear session
keras.backend.clear_session()

print("=" * 60)
print("Bearing Life Prediction using Feed-Forward Neural Network")
print("=" * 60)

# ========== Data Loading ==========
print("\n[1] Loading data...")
data_t5_kur = loadmat('KURTOSIST5.mat')
data_t5_rms = loadmat('RMST5.mat')
data_t5_time = loadmat('timeT5.mat')

# Extract T5 data (adjust keys as needed)
RMS = data_t5_rms['RMS'].flatten()
KUR = data_t5_kur['KURTOSIS'].flatten()
n = len(RMS)
life = np.arange(1, n + 1)
lifePer = np.linspace(0, 1, n)

# Load T6 data for testing
data_t6 = loadmat('time.mat')  # Contains RMS, KURTOSIS, time for T6
RMS6 = data_t6['rmst6'].flatten()[:15]
KUR6 = data_t6['kurtt6'].flatten()[:15]
m = len(RMS6)
life6 = np.arange(1, m + 1)
lifeper6 = np.linspace(0, 1, m)

# Plot original data
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
axes[0].plot(life, lifePer)
axes[0].set_title('%life - life')
axes[0].grid(True)

axes[1].plot(lifePer, RMS, 'b', label='Original')
axes[1].set_title('RMS (blue: Original, red: smooth)')
axes[1].grid(True)

axes[2].plot(lifePer, KUR, 'b', label='Original')
axes[2].set_title('Kurtosis (blue: Original, red: smooth)')
axes[2].grid(True)

# ========== Data Smoothing ==========
print("[2] Smoothing data...")
RMSoriginal = RMS.copy()
KURoriginal = KUR.copy()
smooth_window_size = 5

for ii in range(smooth_window_size - 1, n):
    RMS[ii] = np.mean(RMS[ii - smooth_window_size + 1:ii + 1])
    KUR[ii] = np.mean(KUR[ii - smooth_window_size + 1:ii + 1])

axes[1].plot(lifePer, RMS, 'r', label='Smoothed')
axes[1].legend()
axes[2].plot(lifePer, KUR, 'r', label='Smoothed')
axes[2].legend()
plt.tight_layout()
plt.show()

# ========== Fitting ==========
print("[3] Fitting data with Generalized Weibull function...")
data_for_fit = np.vstack([RMS, KUR])
featureNo = data_for_fit.shape[0]
StartPoints = np.array([[0.078, 1, 5], [3.5, 1, 3]])
Y = np.zeros((n, featureNo))

# Define the fitting function: AA + BB * x^CC
def weibull_func(x, AA, BB, CC):
    return AA + BB * x**CC

for ii in range(featureNo):
    y = data_for_fit[ii, :]
    SP = StartPoints[ii, :]
    
    try:
        popt, _ = curve_fit(weibull_func, life, y, p0=SP, maxfev=10000)
        Y[:, ii] = weibull_func(life, *popt)
        
        # Plot fitting result
        plt.figure(figsize=(8, 5))
        plt.plot(life, y, 'b.', label='Data')
        plt.plot(life, Y[:, ii], 'r-', linewidth=2, label='Fit')
        plt.title(f'Fitting Result {ii + 1}')
        plt.xlabel('Life')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print(f"  Feature {ii + 1} fitted: AA={popt[0]:.4f}, BB={popt[1]:.4f}, CC={popt[2]:.4f}")
    except Exception as e:
        print(f"  Warning: Fitting failed for feature {ii + 1}. Using original data.")
        Y[:, ii] = y

RMSfit = Y[:, 0]
KURfit = Y[:, 1]
RMScomb = RMSfit.copy()
KURcomb = KURfit.copy()

# ========== Data Combining ==========
print("[4] Preparing manual data division...")
ValRatio = 0.3
TestRatio = 0.1
TrainRatio = 1 - ValRatio - TestRatio

# Random division
indices = np.random.permutation(n)
train_size = int(n * TrainRatio)
val_size = int(n * ValRatio)

trainInd = indices[:train_size]
valInd = indices[train_size:train_size + val_size]
testInd = indices[train_size + val_size:]

# Plot divided data
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(RMS, 'b-')
axes[0].plot(trainInd, RMS[trainInd], '.r', label='Train')
axes[0].plot(valInd, RMS[valInd], '.y', label='Val')
axes[0].plot(testInd, RMS[testInd], '.g', label='Test')
axes[0].set_title('RMS Division')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(KUR, 'b-')
axes[1].plot(trainInd, KUR[trainInd], '.r', label='Train')
axes[1].plot(valInd, KUR[valInd], '.y', label='Val')
axes[1].plot(testInd, KUR[testInd], '.g', label='Test')
axes[1].set_title('Kurtosis Division')
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.show()

# Combine data
RMScomb = np.concatenate([RMSfit[trainInd], RMS[valInd], RMS[testInd]])
KURcomb = np.concatenate([KURfit[trainInd], KUR[valInd], KUR[testInd]])
Targ = np.concatenate([lifePer[trainInd], lifePer[valInd], lifePer[testInd]])
Life = np.concatenate([life[trainInd], life[valInd], life[testInd]])

# ========== Data Preparing ==========
print("[5] Preparing learning data...")
# Create time-series features: current and previous time steps
LearningData = np.vstack([
    Life[1:],
    RMScomb[1:],
    KURcomb[1:],
    Life[:-1],
    RMScomb[:-1],
    KURcomb[:-1]
])

TestingData = np.vstack([
    life6[1:],
    RMS6[1:],
    KUR6[1:],
    life6[:-1],
    RMS6[:-1],
    KUR6[:-1]
])

Target = Targ[1:]

print(f"  Learning data shape: {LearningData.shape}")
print(f"  Target shape: {Target.shape}")
print(f"  Testing data shape: {TestingData.shape}")

# ========== Training ==========
print("\n[6] Building and training neural network...")

# Build feedforward network with 5 hidden neurons
model = keras.Sequential([
    layers.Dense(5, activation='tanh', input_shape=(LearningData.shape[0],)),
    layers.Dense(1, activation='linear')
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Manual data splitting using divideblock approach
train_end = int(len(Target) * TrainRatio)
val_end = train_end + int(len(Target) * ValRatio)

X_train = LearningData[:, :train_end].T
y_train = Target[:train_end]
X_val = LearningData[:, train_end:val_end].T
y_val = Target[train_end:val_end]
X_test = LearningData[:, val_end:].T
y_test = Target[val_end:]

# Train the network
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=1
)

# ========== Test ==========
print("\n[7] Testing on T6 data...")
X_test_t6 = TestingData.T
Results = model.predict(X_test_t6).flatten()

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(life6, lifeper6, 'b-', linewidth=2, label='Actual Life Percentage')
plt.plot(life6[1:], Results, 'r--', linewidth=2, label='Predicted')
plt.xlabel('Life (samples)')
plt.ylabel('Life Percentage')
plt.title('Bearing Life Prediction Results')
plt.legend()
plt.grid(True)
plt.show()

# Calculate prediction error
if len(Results) == len(lifeper6[1:]):
    error = np.abs(Results - lifeper6[1:])
    print(f"\nPrediction Error Statistics:")
    print(f"  Mean Absolute Error: {np.mean(error):.6f}")
    print(f"  Max Error: {np.max(error):.6f}")
    print(f"  RMSE: {np.sqrt(np.mean(error**2)):.6f}")

# Save model
model.save('bearing_life_model.h5')
print("\nModel saved as 'bearing_life_model.h5'")
print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)