# Train Custom Model Using Scikit Learn - Deadlift
# 1. Read in Collected Data and Process

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Read the data
df = pd.read_csv('deadlift_with_scaled_angles.csv')

# Display basic info
print("Data shape:", df.shape)
print("\nFirst 3 rows:")
print(df.head(3))

print("\nUnique classes:")
print(df['class'].unique())

print("\nCorrect up samples:")
print(df[df['class'] == 'd_correct_up'])

# Prepare data for training
X = df.drop('class', axis=1)
y = df['class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# 2. Train Machine Learning Classification Model

pipelines = {
    'lr': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

print("\nTraining models...")
fit_models = {}
for algorithm, pipeline in pipelines.items():
    print(f"Training {algorithm}...")
    model = pipeline.fit(X_train, y_train)
    fit_models[algorithm] = model

print("\nModels trained successfully!")
print("Available models:", list(fit_models.keys()))

# Test a prediction
print("\nTesting Ridge Classifier prediction:")
print(fit_models['rc'].predict(X_test)[:5])

# 3. Evaluate and Serialize Model

# Initialize a dictionary to store prediction results
predictions = {}

# Make predictions for each model
for algorithm, model in fit_models.items():
    y_pred = model.predict(X_test)  # Predict on test data
    predictions[algorithm] = y_pred  # Store prediction results

# Print classification report for each model
print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)

for algorithm, y_pred in predictions.items():
    print(f'\n--- {algorithm.upper()} model classification evaluation ---')
    print(classification_report(y_test, y_pred))

# Calculate and display metrics
metrics = {
    'accuracy': {},
    'precision': {},
    'recall': {},
    'f1-score': {}
}

for algorithm, y_pred in predictions.items():
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    metrics['accuracy'][algorithm] = accuracy
    metrics['precision'][algorithm] = precision
    metrics['recall'][algorithm] = recall
    metrics['f1-score'][algorithm] = f1

print("\n" + "="*40)
print("SUMMARY METRICS")
print("="*40)

for metric, values in metrics.items():
    print(f'\n--- {metric.upper()} ---')
    for algorithm, score in values.items():
        print(f'{algorithm}: {score:.4f}')

# Visualize evaluation metrics
print("\nGenerating evaluation visualizations...")

metrics_to_plot = ['precision', 'recall', 'f1-score']

for metric in metrics_to_plot:
    plt.figure(figsize=(12, 6))
    plt.title(f'{metric.capitalize()} Visualization')

    for algorithm, y_pred in predictions.items():
        report = classification_report(y_test, y_pred, output_dict=True)
        metric_score = [report[label][metric] for label in report if label not in ['accuracy', 'macro avg', 'weighted avg']]
        plt.plot(df['class'].unique(), metric_score, label=algorithm, marker='o', linestyle='-')

    plt.xlabel('Class')
    plt.ylabel(metric.capitalize())
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 4. Make Detections with Model

# Save the best performing model (Gradient Boosting)
print(f"\nSaving Gradient Boosting model to 'deadlift.pkl'...")
with open('../model/deadlift.pkl', 'wb') as f:
    pickle.dump(fit_models['gb'], f)

# Also save the scaler separately for reference (optional)
print(f"\nSaving StandardScaler to 'scaler.pkl'...")
scaler = fit_models['gb'].named_steps['standardscaler']
with open('../model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")

# Find the best performing model
best_algorithm = max(metrics['f1-score'], key=metrics['f1-score'].get)
best_score = metrics['f1-score'][best_algorithm]

print(f"\nBest performing model: {best_algorithm.upper()}")
print(f"F1-Score: {best_score:.4f}")

print("\nTraining completed successfully!")