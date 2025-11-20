import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Load the diabetes dataset from Google Drive
dataFrame = pd.read_csv("diabetes.csv")

# Display basic information about the dataset
print(dataFrame.head())
print(dataFrame.info())
print(dataFrame.isnull().sum())
# Separate features and target variable
xFeatures = dataFrame.drop('Outcome', axis=1)
yTarget = dataFrame['Outcome']

# Split into 20/80 split
xTrain, xVal, yTrain, yVal = train_test_split(xFeatures, yTarget, test_size=0.2, random_state=42, stratify=yTarget)

# Standardize features to have mean=0 and variance=1
scaler = StandardScaler()
xTrainScaled = scaler.fit_transform(xTrain)
xValScaled = scaler.transform(xVal)

# Build fully connected neural network with 4 hidden layers
neuralModel = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(xTrainScaled.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model with optimizer, loss function, and metrics
neuralModel.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model architecture summary
print(neuralModel.summary())

# Set up early stopping to prevent overfitting
earlyStopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the neural network model
historyObject = neuralModel.fit(
    xTrainScaled, yTrain,
    epochs=100,
    batch_size=32,
    validation_data=(xValScaled, yVal),
    callbacks=[earlyStopping],
    verbose=1
)

# Plot training and validation loss and accuracy curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(historyObject.history['loss'], label='Training Loss')
plt.plot(historyObject.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(historyObject.history['accuracy'], label='Training Accuracy')
plt.plot(historyObject.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.grid(True)

plt.tight_layout()
plt.show()

# Make predictions with neural network and convert probabilities to binary classes
yPredProba = neuralModel.predict(xValScaled)
yPredNN = (yPredProba > 0.5).astype(int).flatten()

# Calculate performance metrics for neural network
nnAccuracy = accuracy_score(yVal, yPredNN)
nnPrecision = precision_score(yVal, yPredNN)
nnRecall = recall_score(yVal, yPredNN)
nnF1 = f1_score(yVal, yPredNN)
# Display neural network performance metrics
print("Neural Network Performance:")
print(f"Accuracy: {nnAccuracy:.4f}")
print(f"Precision: {nnPrecision:.4f}")
print(f"Recall: {nnRecall:.4f}")
print(f"F1 Score: {nnF1:.4f}")

# Train logistic regression model for comparison
logisticModel = LogisticRegression(max_iter=1000, random_state=42)
logisticModel.fit(xTrainScaled, yTrain)
yPredLR = logisticModel.predict(xValScaled)

# Calculate performance metrics for logistic regression
lrAccuracy = accuracy_score(yVal, yPredLR)
lrPrecision = precision_score(yVal, yPredLR)
lrRecall = recall_score(yVal, yPredLR)
lrF1 = f1_score(yVal, yPredLR)

# Display logistic regression performance metrics
print("\nLogistic Regression Performance:")
print(f"Accuracy: {lrAccuracy:.4f}")
print(f"Precision: {lrPrecision:.4f}")
print(f"Recall: {lrRecall:.4f}")
print(f"F1 Score: {lrF1:.4f}")

# Train support vector machine model for comparison
svmModel = SVC(kernel='rbf', random_state=42)
svmModel.fit(xTrainScaled, yTrain)
yPredSVM = svmModel.predict(xValScaled)
# Calculate performance metrics for SVM
svmAccuracy = accuracy_score(yVal, yPredSVM)
svmPrecision = precision_score(yVal, yPredSVM)
svmRecall = recall_score(yVal, yPredSVM)
svmF1 = f1_score(yVal, yPredSVM)

# Display SVM performance metrics
print("\nSupport Vector Machine Performance:")
print(f"Accuracy: {svmAccuracy:.4f}")
print(f"Precision: {svmPrecision:.4f}")
print(f"Recall: {svmRecall:.4f}")
print(f"F1 Score: {svmF1:.4f}")

# Create comparison table for all three models
comparisonData = {
    'Model': ['Neural Network', 'Logistic Regression', 'SVM'],
    'Accuracy': [nnAccuracy, lrAccuracy, svmAccuracy],
    'Precision': [nnPrecision, lrPrecision, svmPrecision],
    'Recall': [nnRecall, lrRecall, svmRecall],
    'F1 Score': [nnF1, lrF1, svmF1]
}

comparisonDf = pd.DataFrame(comparisonData)
print("\nModel Comparison:")
print(comparisonDf.to_string(index=False))

# Create bar chart comparing all models across all metrics
metricsToPlot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
xPositions = np.arange(len(metricsToPlot))
barWidth = 0.25
plt.figure(figsize=(12, 6))
plt.bar(xPositions - barWidth, [nnAccuracy, nnPrecision, nnRecall, nnF1], barWidth, label='Neural Network')
plt.bar(xPositions, [lrAccuracy, lrPrecision, lrRecall, lrF1], barWidth, label='Logistic Regression')
plt.bar(xPositions + barWidth, [svmAccuracy, svmPrecision, svmRecall, svmF1], barWidth, label='SVM')

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(xPositions, metricsToPlot)
plt.legend()
plt.ylim([0, 1])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()