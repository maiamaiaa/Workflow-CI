"""
Model Training Script for Heart Disease Classification
======================================================
Nama: Eugenia Grasela Maia

Script ini melakukan training model menggunakan data yang sudah
dipreprocessing dari Repository 1.

Features:
- Load preprocessed data
- Train multiple models
- Evaluate and compare models
- Save best model
- Log metrics with MLflow
"""

import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import warnings

warnings.filterwarnings('ignore')


def load_preprocessed_data(data_dir='heart_disease_preprocessing'):
    """
    Load preprocessed data from CSV files.
    
    Args:
        data_dir: Directory containing preprocessed data
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("=" * 60)
    print("Loading Preprocessed Data")
    print("=" * 60)
    
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model and return metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        
    Returns:
        dict: Dictionary of metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1_score': f1_score(y_test, y_pred, average='binary'),
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and compare performance.
    
    Args:
        X_train, X_test, y_train, y_test: Train and test data
        
    Returns:
        tuple: (best_model, best_model_name, all_results)
    """
    print("\n" + "=" * 60)
    print("Training Models")
    print("=" * 60)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
    }
    
    results = {}
    best_model = None
    best_model_name = None
    best_f1 = 0
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, name)
        results[name] = {'model': model, 'metrics': metrics}
        
        # Track best model based on F1-score
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_model = model
            best_model_name = name
    
    print(f"\n{'=' * 60}")
    print(f"Best Model: {best_model_name} (F1-Score: {best_f1:.4f})")
    print(f"{'=' * 60}")
    
    return best_model, best_model_name, results


def log_to_mlflow(results, best_model_name, X_train, y_train):
    """
    Log experiment results to MLflow.
    
    Args:
        results: Dictionary of model results
        best_model_name: Name of the best model
        X_train, y_train: Training data for signature
    """
    print("\n" + "=" * 60)
    print("Logging to MLflow")
    print("=" * 60)
    
    # Set experiment
    mlflow.set_experiment("Heart_Disease_Classification")
    
    for name, result in results.items():
        with mlflow.start_run(run_name=name):
            # Log parameters
            model = result['model']
            params = model.get_params()
            for param_name, param_value in params.items():
                if param_value is not None and not callable(param_value):
                    try:
                        mlflow.log_param(param_name, param_value)
                    except:
                        pass
            
            # Log metrics
            for metric_name, metric_value in result['metrics'].items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            if name == best_model_name:
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name="HeartDiseaseModel"
                )
                mlflow.set_tag("best_model", "true")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            mlflow.set_tag("model_type", name)
            
            print(f"  Logged: {name}")
    
    print("MLflow logging complete!")


def save_best_model(model, model_name, output_dir='model'):
    """
    Save the best model to disk.
    
    Args:
        model: Trained model to save
        model_name: Name of the model
        output_dir: Output directory
    """
    print("\n" + "=" * 60)
    print("Saving Best Model")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = f'{output_dir}/best_model.pkl'
    joblib.dump(model, model_path)
    
    # Save model info
    model_info = {
        'model_name': model_name,
        'model_type': type(model).__name__,
        'model_params': model.get_params()
    }
    
    info_path = f'{output_dir}/model_info.txt'
    with open(info_path, 'w') as f:
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Model saved to: {model_path}")
    print(f"Model info saved to: {info_path}")


def print_summary(results, best_model_name):
    """
    Print training summary.
    
    Args:
        results: Dictionary of model results
        best_model_name: Name of the best model
    """
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    # Create comparison table
    print(f"\n{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
    print("-" * 75)
    
    for name, result in results.items():
        metrics = result['metrics']
        roc_auc = metrics.get('roc_auc', 0)
        marker = " *" if name == best_model_name else ""
        print(f"{name:<25} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} {roc_auc:<10.4f}{marker}")
    
    print("-" * 75)
    print("* = Best Model")


def main():
    """
    Main function to run the complete training pipeline.
    """
    print("=" * 60)
    print("HEART DISEASE MODEL TRAINING PIPELINE")
    print("Author: Eugenia Grasela Maia")
    print("=" * 60)
    
    # Configuration
    DATA_DIR = 'heart_disease_preprocessing'
    MODEL_DIR = 'model'
    
    # Step 1: Load preprocessed data
    X_train, X_test, y_train, y_test = load_preprocessed_data(DATA_DIR)
    
    # Step 2: Train and evaluate models
    best_model, best_model_name, results = train_models(
        X_train, X_test, y_train, y_test
    )
    
    # Step 3: Log to MLflow
    log_to_mlflow(results, best_model_name, X_train, y_train)
    
    # Step 4: Save best model
    save_best_model(best_model, best_model_name, MODEL_DIR)
    
    # Step 5: Print summary
    print_summary(results, best_model_name)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nBest Model: {best_model_name}")
    print(f"Model saved to: {MODEL_DIR}/best_model.pkl")
    print("MLflow experiment: Heart_Disease_Classification")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
