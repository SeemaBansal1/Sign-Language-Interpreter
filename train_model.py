"""
Model Training Script for Sign Language Recognition
===================================================
This script trains a Random Forest classifier on the collected hand landmark data.

Usage:
    python train_model.py

The script will:
    1. Load data from data.csv
    2. Preprocess and split the data
    3. Train a Random Forest classifier
    4. Evaluate the model
    5. Save the model to model.p
"""

import pickle
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


class SignLanguageModelTrainer:
    def __init__(self, data_file="data.csv", model_file="model.p"):
        """Initialize the trainer."""
        self.data_file = data_file
        self.model_file = model_file
        self.model = None
        self.label_encoder = LabelEncoder()
        self.classes = None
    
    def load_data(self):
        """Load and validate the training data."""
        print("\n" + "=" * 50)
        print("Loading Data")
        print("=" * 50)
        
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(
                f"Data file '{self.data_file}' not found!\n"
                "Please run collect_data.py first to collect training data."
            )
        
        # Load CSV
        df = pd.read_csv(self.data_file)
        print(f"Loaded {len(df)} samples from {self.data_file}")
        
        # Validate data
        if 'label' not in df.columns:
            raise ValueError("CSV must have a 'label' column!")
        
        # Check for expected number of features (63 = 21 landmarks * 3 coordinates)
        expected_features = 63
        actual_features = len(df.columns) - 1  # Exclude label column
        
        if actual_features != expected_features:
            print(f"Warning: Expected {expected_features} features, got {actual_features}")
        
        # Display label distribution
        print("\nLabel Distribution:")
        print("-" * 30)
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count} samples")
        
        # Separate features and labels
        X = df.drop('label', axis=1).values
        y = df['label'].values
        
        # Handle any missing values
        if np.isnan(X).any():
            print("\nWarning: Found NaN values, replacing with 0")
            X = np.nan_to_num(X, 0)
        
        return X, y
    
    def preprocess_data(self, X, y, test_size=0.2, random_state=42):
        """Preprocess and split the data."""
        print("\n" + "=" * 50)
        print("Preprocessing Data")
        print("=" * 50)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes = self.label_encoder.classes_
        
        print(f"Classes: {list(self.classes)}")
        print(f"Number of classes: {len(self.classes)}")
        print(f"Feature dimensions: {X.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_encoded
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        """Train the Random Forest classifier."""
        print("\n" + "=" * 50)
        print("Training Model")
        print("=" * 50)
        
        # Create Random Forest classifier with optimized parameters
        self.model = RandomForestClassifier(
            n_estimators=100,           # Number of trees
            max_depth=20,               # Maximum depth of trees
            min_samples_split=5,        # Minimum samples to split
            min_samples_leaf=2,         # Minimum samples at leaf
            max_features='sqrt',        # Features to consider at split
            bootstrap=True,             # Use bootstrap samples
            n_jobs=-1,                  # Use all CPU cores
            random_state=42,            # Reproducibility
            class_weight='balanced'     # Handle class imbalance
        )
        
        print("Training Random Forest classifier...")
        print("Parameters:")
        print(f"  - n_estimators: 100")
        print(f"  - max_depth: 20")
        print(f"  - class_weight: balanced")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Cross-validation
        print("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate the trained model."""
        print("\n" + "=" * 50)
        print("Model Evaluation")
        print("=" * 50)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        
        # Classification report
        print("\nClassification Report:")
        print("-" * 50)
        target_names = [str(c) for c in self.classes]
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        print("Confusion Matrix:")
        print("-" * 50)
        cm = confusion_matrix(y_test, y_pred)
        
        # Pretty print confusion matrix
        print(f"{'':>12}", end='')
        for name in target_names:
            print(f"{name:>10}", end='')
        print()
        
        for i, row in enumerate(cm):
            print(f"{target_names[i]:>12}", end='')
            for val in row:
                print(f"{val:>10}", end='')
            print()
        
        # Feature importance (top 10)
        print("\nTop 10 Important Features:")
        print("-" * 50)
        
        feature_importance = self.model.feature_importances_
        indices = np.argsort(feature_importance)[::-1][:10]
        
        for i, idx in enumerate(indices):
            landmark_idx = idx // 3
            coord = ['x', 'y', 'z'][idx % 3]
            print(f"  {i+1}. Landmark {landmark_idx} ({coord}): {feature_importance[idx]:.4f}")
        
        return accuracy
    
    def save_model(self):
        """Save the trained model and label encoder."""
        print("\n" + "=" * 50)
        print("Saving Model")
        print("=" * 50)
        
        model_data = {
            'model': self.model,
            'labels': self.classes,
            'label_encoder': self.label_encoder
        }
        
        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {self.model_file}")
        print(f"Classes saved: {list(self.classes)}")
        
        # Verify saved model
        file_size = os.path.getsize(self.model_file) / 1024
        print(f"Model file size: {file_size:.2f} KB")
    
    def run(self):
        """Run the complete training pipeline."""
        print("\n" + "=" * 60)
        print("  SIGN LANGUAGE MODEL TRAINING")
        print("=" * 60)
        
        try:
            # Load data
            X, y = self.load_data()
            
            # Check minimum samples
            if len(X) < 10:
                print("\nWarning: Very few samples! Consider collecting more data.")
                print("Recommended: At least 50 samples per class")
            
            # Preprocess
            X_train, X_test, y_train, y_test = self.preprocess_data(X, y)
            
            # Train
            self.train(X_train, y_train)
            
            # Evaluate
            accuracy = self.evaluate(X_test, y_test)
            
            # Save
            self.save_model()
            
            print("\n" + "=" * 60)
            print("  TRAINING COMPLETE!")
            print("=" * 60)
            print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
            print(f"Model saved to: {self.model_file}")
            print("\nNext steps:")
            print("  1. Run 'python app.py' to start the web application")
            print("  2. Open http://localhost:5000 in your browser")
            print("  3. Show hand signs to the camera!")
            
            return accuracy
            
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            return None
        except Exception as e:
            print(f"\nError during training: {e}")
            raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Sign Language Recognition Model')
    parser.add_argument('--data', type=str, default='data.csv',
                        help='Path to training data CSV (default: data.csv)')
    parser.add_argument('--model', type=str, default='model.p',
                        help='Path to save model (default: model.p)')
    
    args = parser.parse_args()
    
    trainer = SignLanguageModelTrainer(
        data_file=args.data,
        model_file=args.model
    )
    
    trainer.run()


if __name__ == "__main__":
    main()
