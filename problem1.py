import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

#diabetes
dataFrame = pd.read_csv("diabetes.csv")

print(dataFrame.head())
print(dataFrame.info())
print(dataFrame.isnull().sum())

xFeatures = dataFrame.drop('Outcome', axis=1)
yTarget = dataFrame['Outcome']

#80/20
xTrain, xVal, yTrain, yVal = train_test_split(
    xFeatures, yTarget, test_size=0.2, random_state=42, stratify=yTarget
)
scaler = StandardScaler()
xTrainScaled = scaler.fit_transform(xTrain)
xValScaled = scaler.transform(xVal)

neuralModel = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(xTrainScaled.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Outputting 0/1
])

#compile the model with adam
neuralModel.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(neuralModel.summary())

earlyStopping = EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True
)

#trian
historyObject = neuralModel.fit(
    xTrainScaled, yTrain,
    epochs=100,
    batch_size=32,
    validation_data=(xValScaled, yVal),
    callbacks=[earlyStopping],
    verbose=1
)

#Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(historyObject.history['loss'], label='Training Loss')
plt.plot(historyObject.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss (Please behave)')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(historyObject.history['accuracy'], label='Training Accuracy')
plt.plot(historyObject.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# convert probability blobs to 0 or 1.
yPredNN = (neuralModel.predict(xValScaled) > 0.5).astype(int).flatten()

# Function to compute all metric
def get_metrics(true, pred):
    return (
        accuracy_score(true, pred),
        precision_score(true, pred),
        recall_score(true, pred),
        f1_score(true, pred)
    )

# Evaluate DNN
nnAccuracy, nnPrecision, nnRecall, nnF1 = get_metrics(yVal, yPredNN)

print("Neural Network Performance:")
print(f"Accuracy: {nnAccuracy:.4f}")
print(f"Precision: {nnPrecision:.4f}")
print(f"Recall: {nnRecall:.4f}")
print(f"F1 Score: {nnF1:.4f}")

# Logistic Regression
logisticModel = LogisticRegression(max_iter=1000, random_state=42)
logisticModel.fit(xTrainScaled, yTrain)
yPredLR = logisticModel.predict(xValScaled)
lrAccuracy, lrPrecision, lrRecall, lrF1 = get_metrics(yVal, yPredLR)

print("\nLogistic Regression Performance:")
print(f"Accuracy: {lrAccuracy:.4f}")
print(f"Precision: {lrPrecision:.4f}")
print(f"Recall: {lrRecall:.4f}")
print(f"F1 Score: {lrF1:.4f}")

# SVM
svmModel = SVC(kernel='rbf', random_state=42)
svmModel.fit(xTrainScaled, yTrain)
yPredSVM = svmModel.predict(xValScaled)
svmAccuracy, svmPrecision, svmRecall, svmF1 = get_metrics(yVal, yPredSVM)

print("\nSupport Vector Machine Performance:")
print(f"Accuracy: {svmAccuracy:.4f}")
print(f"Precision: {svmPrecision:.4f}")
print(f"Recall: {svmRecall:.4f}")
print(f"F1 Score: {svmF1:.4f}")

#Table 
comparisonDf = pd.DataFrame({
    'Model': ['Neural Network', 'Logistic Regression', 'SVM'],
    'Accuracy': [nnAccuracy, lrAccuracy, svmAccuracy],
    'Precision': [nnPrecision, lrPrecision, svmPrecision],
    'Recall': [nnRecall, lrRecall, svmRecall],
    'F1 Score': [nnF1, lrF1, svmF1]
})

print("\nModel Comparison:")
print(comparisonDf.to_string(index=False))

# Bar chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
x = np.arange(len(metrics))
barWidth = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - barWidth, comparisonDf.iloc[0, 1:], barWidth, label='Neural Net')
plt.bar(x, comparisonDf.iloc[1, 1:], barWidth, label='Log Reg')
plt.bar(x + barWidth, comparisonDf.iloc[2, 1:], barWidth, label='SVM')

plt.xticks(x, metrics)
plt.ylabel('Score')
plt.title('Model Comparison (aka: which one disappointed us the least)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])
plt.tight_layout()
plt.show()
