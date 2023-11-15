import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load saved validation predictions
with open('./output/val_predictions_normalized.pkl', 'rb') as f:
    val_predictions_normalized = pickle.load(f)
with open('./output/val_predictions_non_normalized.pkl', 'rb') as f:
    val_predictions_non_normalized = pickle.load(f)

# Generate x-coordinates starting from 500
x_coordinates = np.arange(500, len(val_predictions_normalized))

plt.figure(figsize=(12, 6))

# Plot predictions with normalization/standardization
plt.plot(x_coordinates, val_predictions_normalized[500:], label='With Normalization', color='blue')

# Plot predictions without normalization/standardization
plt.plot(x_coordinates, val_predictions_non_normalized[500:], label='Without Normalization', color='red')

plt.title('Validation Data Predictions with and without Normalization/Standardization')
plt.xlabel('Data Points')
plt.ylabel('Predicted Value')
plt.legend()
plt.show()
