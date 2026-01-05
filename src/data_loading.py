"""
Data loading and preprocessing module for XAI analysis.

This module handles loading the Iris dataset and preparing it for model training
and explanation analysis.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_iris_data():
    """
    Load the Iris dataset and prepare it for analysis.
    
    Returns:
        tuple: A tuple containing:
            - X (pd.DataFrame): Feature data
            - y (pd.Series): Target labels
            - feature_names (list): List of feature names
            - class_names (list): List of class names
            - df (pd.DataFrame): Combined dataframe with features, target, and class names
    """
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    y = iris.target.copy()
    feature_names = list(X.columns)
    class_names = list(iris.target_names)
    
    df = X.copy()
    df["target"] = y
    df["class_name"] = df["target"].map(lambda t: class_names[t])
    
    return X, y, feature_names, class_names, df


def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Feature data
        y (pd.Series): Target labels
        test_size (float): Proportion of dataset to include in test split
        random_state (int): Random seed for reproducibility
        stratify (bool): Whether to stratify the split based on target distribution
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    return X_train, X_test, y_train, y_test

