import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import time

# Load CIFAR-10 dataset
(xTrainRaw, yTrainRaw), (xTestRaw, yTestRaw) = cifar10.load_data()

# Display dataset information
print(f"Training samples: {xTrainRaw.shape[0]}")
print(f"Test samples: {xTestRaw.shape[0]}")
print(f"Image shape: {xTrainRaw.shape[1:]}")

# Normalize pixel values to range [0, 1]
xTrain = xTrainRaw.astype('float32') / 255.0
xTest = xTestRaw.astype('float32') / 255.0

# Flatten images from 32x32x3 to 3072-dimensional vectors
xTrainFlat = xTrain.reshape(xTrain.shape[0], -1)
xTestFlat = xTest.reshape(xTest.shape[0], -1)

# Convert labels to one-hot encoded vectors
yTrainOneHot = to_categorical(yTrainRaw, 10)
yTestOneHot = to_categorical(yTestRaw, 10)

print(f"Flattened training shape: {xTrainFlat.shape}")
print(f"One-hot encoded labels shape: {yTrainOneHot.shape}")

# Build fully connected neural network with three hidden layers
deeperModel = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(3072,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model with Adam optimizer and categorical crossentropy loss
deeperModel.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture and count parameters
print("\nDeeper Model Architecture:")
print(deeperModel.summary())

# Count total parameters in the model
totalParams = deeperModel.count_params()
print(f"\nTotal Parameters in Deeper Model: {totalParams:,}")

# Record start time for training
startTime = time.time()

# Train the model for 300 epochs and store history
historyObject = deeperModel.fit(
    xTrainFlat, yTrainOneHot,
    epochs=300,
    batch_size=128,
    validation_data=(xTestFlat, yTestOneHot),
    verbose=1
)

# Calculate total training time
endTime = time.time()
totalTrainingTime = endTime - startTime

print(f"\nTotal Training Time: {totalTrainingTime:.2f} seconds ({totalTrainingTime/60:.2f} minutes)")

# Extract training metrics for each epoch
epochNumbers = list(range(1, len(historyObject.history['loss']) + 1))
trainingLosses = historyObject.history['loss']
trainingAccuracies = historyObject.history['accuracy']
validationLosses = historyObject.history['val_loss']
validationAccuracies = historyObject.history['val_accuracy']

# Display results at key epochs
print("\nResults at Key Epochs:")
print("Epoch | Train Loss | Train Acc | Val Loss | Val Acc")
print("-" * 60)
keyEpochs = [1, 10, 50, 100, 150, 200, 250, 300]
for epoch in keyEpochs:
    if epoch <= len(epochNumbers):
        idx = epoch - 1
        print(f"{epoch:5d} | {trainingLosses[idx]:10.4f} | {trainingAccuracies[idx]:9.4f} | {validationLosses[idx]:8.4f} | {validationAccuracies[idx]:7.4f}")

# Evaluate model on test set
testLoss, testAccuracy = deeperModel.evaluate(xTestFlat, yTestOneHot, verbose=0)
print(f"\nFinal Test Accuracy after 300 epochs: {testAccuracy:.4f}")
print(f"Final Test Loss after 300 epochs: {testLoss:.4f}")

# Plot training and validation loss
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochNumbers, trainingLosses, label='Training Loss', linewidth=2)
plt.plot(epochNumbers, validationLosses, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (300 Epochs)')
plt.legend()
plt.grid(True)
# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochNumbers, trainingAccuracies, label='Training Accuracy', linewidth=2)
plt.plot(epochNumbers, validationAccuracies, label='Validation Accuracy', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy (300 Epochs)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Create summary statistics
print("\nTraining Summary:")
print(f"Best Training Accuracy: {max(trainingAccuracies):.4f} at Epoch {trainingAccuracies.index(max(trainingAccuracies)) + 1}")
print(f"Best Validation Accuracy: {max(validationAccuracies):.4f} at Epoch {validationAccuracies.index(max(validationAccuracies)) + 1}")
print(f"Final Training Loss: {trainingLosses[-1]:.4f}")
print(f"Final Validation Loss: {validationLosses[-1]:.4f}")

# Calculate overfitting metrics
trainingValGap = trainingAccuracies[-1] - validationAccuracies[-1]
lossGap = validationLosses[-1] - trainingLosses[-1]
print(f"\nOverfitting Analysis:")
print(f"Training-Validation Accuracy Gap: {trainingValGap:.4f} ({trainingValGap*100:.2f}%)")
print(f"Validation-Training Loss Gap: {lossGap:.4f}")

# Compare with baseline model from Problem 3a
baselineParams = 512 * 3072 + 512 + 10 * 512 + 10
deeperParams = totalParams
print(f"\nModel Comparison:")
print(f"Baseline Model (1 hidden layer) Parameters: {baselineParams:,}")
print(f"Deeper Model (3 hidden layers) Parameters: {deeperParams:,}")
print(f"Parameter Increase: {deeperParams - baselineParams:,} ({((deeperParams/baselineParams - 1) * 100):.2f}%)")
print(f"\nBaseline Model Test Accuracy (50 epochs): 0.5068")
print(f"Deeper Model Test Accuracy (300 epochs): {testAccuracy:.4f}")
print(f"Accuracy Improvement: {(testAccuracy - 0.5068):.4f} ({((testAccuracy - 0.5068) * 100):.2f}%)")