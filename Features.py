import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load the .mat file
data = loadmat('matlab2_2_v1x.mat')

# Extract the xy array (adjust key name if needed)
xy = data['xy']

# Initialize arrays to store results
num_columns = xy.shape[1]
KURTOSIS = np.zeros(num_columns)
RMS = np.zeros(num_columns)

# Loop through each column
for i in range(num_columns):
    X = xy[:, i]
    
    # Calculate mean
    me = np.mean(X)
    
    # Calculate variance (v2)
    v2 = np.sum((X - np.mean(X))**2) / len(X)
    
    # Calculate fourth moment (b)
    b = np.sum((X - np.mean(X))**4) / len(X)
    
    # Calculate kurtosis factor (k)
    k = b / (v2**2)
    
    KURTOSIS[i] = k
    
    # Calculate RMS (root mean square)
    rms = np.sqrt(np.sum(X**2) / len(X))
    RMS[i] = rms

# Display results
print(f"Calculated {num_columns} samples")
print(f"\nKurtosis shape: {KURTOSIS.shape}")
print(f"RMS shape: {RMS.shape}")
print(f"\nKurtosis range: [{KURTOSIS.min():.4f}, {KURTOSIS.max():.4f}]")
print(f"RMS range: [{RMS.min():.4f}, {RMS.max():.4f}]")

# Optional: Plot RMS
plt.figure(figsize=(10, 6))
plt.plot(RMS, marker='o', linestyle='-', markersize=3)
plt.grid(True)
plt.legend(['RMS'])
plt.title('Trend of RMS')
plt.xlabel('Number of Data')
plt.ylabel('Amplitude of RMS')
plt.show()

# Optional: Save results
# np.save('KURTOSIS.npy', KURTOSIS)
# np.save('RMS.npy', RMS)