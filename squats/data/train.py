import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to Python path to find utils package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    precision_score, accuracy_score, f1_score, recall_score, 
    confusion_matrix, roc_curve, auc
)

import warnings
warnings.filterwarnings('ignore')

# Import from utils package
from utils import describe_dataset, round_up_metric_results


class SquatModelTrainer:
    """Class for training squat pose classification models"""
    
    def __init__(self, train_csv_path="./train.csv", test_csv_path="./test.csv"):
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Define algorithms to test (will be adjusted based on data size)
        self.base_algorithms = [
            ("LR", LogisticRegression()),
            ("SVC", SVC(probability=True)),
            ('KNN', KNeighborsClassifier()),
            ("DTC", DecisionTreeClassifier()),
            ("SGDC", CalibratedClassifierCV(SGDClassifier())),
            ("NB", GaussianNB()),
            ('RF', RandomForestClassifier()),
        ]
        self.algorithms = []
    
    def load_and_preprocess_data(self):
        """Load and preprocess training data"""
        print("Loading and preprocessing data...")
        
        # Load training data
        df = describe_dataset(self.train_csv_path)
        
        # Check if we have enough data
        if len(df) < 10:
            print(f"Warning: Very small dataset ({len(df)} samples). Results may not be reliable.")
            print("Consider collecting more training data for better model performance.")
        
        # Visualize class distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(x='label', data=df, palette="Set1")
        plt.title("Class Distribution in Training Data")
        plt.show()
        
        # Convert labels to numeric
        df.loc[df["label"] == "down", "label"] = 0
        df.loc[df["label"] == "up", "label"] = 1
        
        # Extract features and labels
        X = df.drop("label", axis=1)  # features
        y = df["label"].astype("int")  # labels
        
        # Scale features
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X))
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=1234
        )
        
        # Adjust algorithms based on training data size
        self._adjust_algorithms_for_data_size()
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
        return df
    
    def _adjust_algorithms_for_data_size(self):
        """Adjust algorithm parameters based on available training data size"""
        n_samples = len(self.X_train)
        
        print(f"Adjusting algorithms for {n_samples} training samples...")
        
        # Clear existing algorithms
        self.algorithms = []
        
        # Add algorithms with appropriate parameters
        for name, model in self.base_algorithms:
            if name == 'KNN':
                # Adjust KNN neighbors based on sample size
                # Use at most n_samples-1 neighbors, but at least 1
                k_neighbors = min(max(1, n_samples - 1), 5)
                if k_neighbors < 5:
                    print(f"Warning: Using {k_neighbors} neighbors for KNN due to small dataset")
                adjusted_model = KNeighborsClassifier(n_neighbors=k_neighbors)
                self.algorithms.append((name, adjusted_model))
            elif name == 'RF':
                # Adjust Random Forest parameters for small datasets
                if n_samples < 10:
                    # Use fewer trees and simpler parameters for very small datasets
                    adjusted_model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
                    print("Warning: Using simplified Random Forest due to small dataset")
                else:
                    adjusted_model = RandomForestClassifier(random_state=42)
                self.algorithms.append((name, adjusted_model))
            elif name == 'SVC':
                # Adjust SVC for small datasets
                if n_samples < 10:
                    adjusted_model = SVC(probability=True, C=1.0, gamma='scale')
                else:
                    adjusted_model = SVC(probability=True)
                self.algorithms.append((name, adjusted_model))
            elif name == 'SGDC':
                # Adjust SGDC cross-validation for small datasets
                if n_samples < 5:
                    # Skip SGDC for very small datasets or use simple SGD without calibration
                    print("Warning: Skipping SGDC due to insufficient data for cross-validation")
                    continue
                else:
                    # Adjust CV folds based on sample size
                    cv_folds = min(3, n_samples)
                    adjusted_model = CalibratedClassifierCV(SGDClassifier(), cv=cv_folds)
                    if cv_folds < 5:
                        print(f"Warning: Using {cv_folds}-fold CV for SGDC due to small dataset")
                    self.algorithms.append((name, adjusted_model))
            else:
                # Use default parameters for other algorithms
                self.algorithms.append((name, model))
    
    def train_models(self):
        """Train multiple ML models and evaluate them"""
        print("Training multiple models...")
        
        final_results = []
        
        for name, model in self.algorithms:
            print(f"Training {name}...")
            
            # Train model
            trained_model = model.fit(self.X_train, self.y_train)
            self.models[name] = trained_model
            
            # Evaluate model
            model_results = model.predict(self.X_test)
            
            # Calculate metrics
            p_score = precision_score(self.y_test, model_results, average=None, labels=[0, 1])
            a_score = accuracy_score(self.y_test, model_results)
            r_score = recall_score(self.y_test, model_results, average=None, labels=[0, 1])
            f1_score_result = f1_score(self.y_test, model_results, average=None, labels=[0, 1])
            cm = confusion_matrix(self.y_test, model_results, labels=[0, 1])
            
            final_results.append((
                name, 
                round_up_metric_results(p_score), 
                a_score, 
                round_up_metric_results(r_score), 
                round_up_metric_results(f1_score_result), 
                cm
            ))
        
        # Sort results by F1 score
        final_results.sort(key=lambda k: sum(k[4]), reverse=True)
        
        # Create results dataframe
        results_df = pd.DataFrame(
            final_results, 
            columns=["Model", "Precision Score", "Accuracy Score", "Recall Score", "F1 Score", "Confusion Matrix"]
        )
        
        print("\nModel Training Results:")
        print(results_df)
        
        return results_df
    
    def evaluate_on_test_set(self):
        """Evaluate models on separate test set"""
        if not os.path.exists(self.test_csv_path):
            print(f"Test dataset not found at {self.test_csv_path}")
            return None
            
        print("Evaluating on separate test set...")
        
        # Load test data
        test_df = describe_dataset(self.test_csv_path)
        test_df.loc[test_df["label"] == "down", "label"] = 0
        test_df.loc[test_df["label"] == "up", "label"] = 1
        
        test_x = test_df.drop("label", axis=1)
        test_y = test_df["label"].astype("int")
        
        # Scale test data using fitted scaler
        test_x_scaled = pd.DataFrame(self.scaler.transform(test_x))
        
        testset_final_results = []
        
        for name, model in self.models.items():
            # Evaluate model
            model_results = model.predict(test_x_scaled)
            
            # Calculate metrics
            p_score = precision_score(test_y, model_results, average="weighted")
            a_score = accuracy_score(test_y, model_results)
            r_score = recall_score(test_y, model_results, average="weighted")
            f1_score_result = f1_score(test_y, model_results, average="weighted")
            cm = confusion_matrix(test_y, model_results, labels=[0, 1])
            
            testset_final_results.append((
                name, p_score, a_score, r_score, f1_score_result, cm
            ))
        
        # Sort by F1 score
        testset_final_results.sort(key=lambda k: k[4], reverse=True)
        
        eval_df = pd.DataFrame(
            testset_final_results, 
            columns=["Model", "Precision Score", "Accuracy Score", "Recall Score", "F1 Score", "Confusion Matrix"]
        )
        eval_df = eval_df.sort_values(by=['F1 Score'], ascending=False).reset_index(drop=True)
        
        # Save evaluation results
        eval_df.to_csv("evaluation.csv", sep=',', encoding='utf-8', index=False)
        
        print("\nTest Set Evaluation Results:")
        print(eval_df)
        
        # Set best model
        best_model_name = eval_df.iloc[0]["Model"]
        self.best_model = self.models[best_model_name]
        print(f"\nBest model: {best_model_name}")
        
        return eval_df, test_x_scaled, test_y
    
    def save_models(self, model_dir="../model"):
        """Save trained models"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save all models
        with open(f"{model_dir}/sklearn_models.pkl", "wb") as f:
            pickle.dump(self.models, f)
        
        # Save individual models
        for name, model in self.models.items():
            with open(f"{model_dir}/{name}_model.pkl", "wb") as f:
                pickle.dump(model, f)
        
        # Save scaler
        with open(f"{model_dir}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        
        print(f"Models saved to {model_dir}")
    
    def plot_confusion_matrix(self, test_x, test_y):
        """Plot confusion matrix for best model"""
        if self.best_model is None:
            print("No best model found. Run evaluate_on_test_set first.")
            return
        
        y_predictions = self.best_model.predict(test_x)
        cm = confusion_matrix(test_y, y_predictions, labels=[0, 1])
        
        cm_df = pd.DataFrame(cm, index=["down", "up"], columns=["down", "up"])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, linewidths=1, annot=True, fmt='g', cmap="crest")
        plt.title("Confusion Matrix - Best Model")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()
    
    def plot_roc_curve(self, test_x, test_y):
        """Plot ROC curve for best model"""
        if self.best_model is None:
            print("No best model found. Run evaluate_on_test_set first.")
            return
        
        # Get prediction probabilities
        probs = self.best_model.predict_proba(test_x)
        preds = probs[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, threshold = roc_curve(test_y, preds)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = threshold[optimal_idx]
        print(f"Optimal Threshold: {optimal_threshold:.3f}")
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'r--', label="Random Guess")
        plt.legend(loc=4)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('ROC Curve - Best Model')
        plt.show()
        
        return optimal_threshold


def main():
    """Main training function"""
    print("Squat Pose Classification Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = SquatModelTrainer()
    
    # Check if training data exists
    if not os.path.exists(trainer.train_csv_path):
        print(f"Training data not found at {trainer.train_csv_path}")
        print("Please run data_collection.py first to collect training data.")
        return
    
    try:
        # Load and preprocess data
        df = trainer.load_and_preprocess_data()
        
        # Train models
        results_df = trainer.train_models()
        
        # Evaluate on test set
        eval_df, test_x, test_y = trainer.evaluate_on_test_set()
        
        # Save models
        trainer.save_models()
        
        # Plot visualizations
        if eval_df is not None:
            trainer.plot_confusion_matrix(test_x, test_y)
            optimal_threshold = trainer.plot_roc_curve(test_x, test_y)
        
        print("\nTraining completed successfully!")
        print(f"Best model: {trainer.best_model}")
        
    except Exception as e:
        print(f"Error during training: {e}")


if __name__ == "__main__":
    main()
