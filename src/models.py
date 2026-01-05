"""
Model training and evaluation module.

This module handles training and evaluating Logistic Regression and Random Forest
classifiers for the Iris dataset.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_logistic_regression(X_train, y_train, random_state=42, max_iter=200):
    """
    Train a Logistic Regression model with StandardScaler preprocessing.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        random_state (int): Random seed for reproducibility
        max_iter (int): Maximum number of iterations
    
    Returns:
        Pipeline: Trained Logistic Regression pipeline
    """
    lr_clf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=max_iter, random_state=random_state))
    ])
    
    lr_clf.fit(X_train, y_train)
    return lr_clf


def train_random_forest(X_train, y_train, n_estimators=300, random_state=42):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        n_estimators (int): Number of trees in the forest
        random_state (int): Random seed for reproducibility
    
    Returns:
        RandomForestClassifier: Trained Random Forest model
    """
    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    
    rf_clf.fit(X_train, y_train)
    return rf_clf


def evaluate_models(lr_clf, rf_clf, X_test, y_test, class_names):
    """
    Evaluate both models and print classification reports.
    
    Args:
        lr_clf: Trained Logistic Regression model
        rf_clf: Trained Random Forest model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        class_names (list): List of class names
    
    Returns:
        tuple: (lr_accuracy, rf_accuracy, lr_predictions, rf_predictions)
    """
    lr_pred = lr_clf.predict(X_test)
    rf_pred = rf_clf.predict(X_test)
    
    lr_accuracy = accuracy_score(y_test, lr_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print("Accuracy LR:", lr_accuracy)
    print("Accuracy RF:", rf_accuracy)
    print("\nClassification Report - Logistic Regression:")
    print(classification_report(y_test, lr_pred, target_names=class_names))
    print("\nClassification Report - Random Forest:")
    print(classification_report(y_test, rf_pred, target_names=class_names))
    
    return lr_accuracy, rf_accuracy, lr_pred, rf_pred


def find_misclassified_instance(lr_clf, rf_clf, X_test, y_test, start_idx=1):
    """
    Find the first misclassified instance (where models disagree or make errors).
    
    Args:
        lr_clf: Trained Logistic Regression model
        rf_clf: Trained Random Forest model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        start_idx (int): Starting index to search from
    
    Returns:
        int or None: Index of misclassified instance, or None if not found
    """
    y_test_list = y_test.tolist()
    for i in range(start_idx, len(X_test)):
        true_class = y_test_list[i]
        current_x_test = X_test.iloc[[i]]
        lr_pred = lr_clf.predict(current_x_test)[0]
        rf_pred = rf_clf.predict(current_x_test)[0]
        if lr_pred != true_class or rf_pred != true_class:
            return i
    return None

