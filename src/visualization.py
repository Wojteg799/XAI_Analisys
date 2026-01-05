"""
Visualization utilities for XAI analysis.

This module provides helper functions for creating plots and visualizations
used in global and local explanation analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_distributions(X, figsize=(10, 6), save_path=None):
    """
    Plot histograms of feature distributions.
    
    Args:
        X (pd.DataFrame): Feature data
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
    """
    X.hist(figsize=figsize)
    plt.suptitle("Feature Distributions in Iris Dataset")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_importance_bar(importance_series, title, ylabel, figsize=(8, 4), save_path=None):
    """
    Plot a bar chart of feature importance.
    
    Args:
        importance_series (pd.Series): Feature importance values
        title (str): Plot title
        ylabel (str): Y-axis label
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
    """
    importance_series.plot(kind="bar", figsize=figsize)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_permutation_importance_comparison(perm_lr, perm_rf, figsize=(12, 4), save_path=None):
    """
    Plot permutation importance for both models side by side.
    
    Args:
        perm_lr (pd.Series): Permutation importance for Logistic Regression
        perm_rf (pd.Series): Permutation importance for Random Forest
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    perm_lr.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Permutation Importance (LR)")
    axes[0].set_ylabel("Accuracy drop after feature shuffling")
    
    perm_rf.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Permutation Importance (RF)")
    axes[1].set_ylabel("Accuracy drop after feature shuffling")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_shap_global_comparison(shap_imp_lr, shap_imp_rf, figsize=(12, 4), save_path=None):
    """
    Plot global SHAP importance for both models side by side.
    
    Args:
        shap_imp_lr (pd.Series): SHAP importance for Logistic Regression
        shap_imp_rf (pd.Series): SHAP importance for Random Forest
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    shap_imp_lr.plot(kind="bar", ax=axes[0])
    axes[0].set_title("SHAP: mean(|value|) – LR (global)")
    axes[0].set_ylabel("mean |SHAP|")
    
    shap_imp_rf.plot(kind="bar", ax=axes[1])
    axes[1].set_title("SHAP: mean(|value|) – RF (global)")
    axes[1].set_ylabel("mean |SHAP|")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_normalized_heatmap(importance_df, title, figsize=(12, 6), save_path=None):
    """
    Plot a normalized heatmap of feature importance across different methods.
    
    Args:
        importance_df (pd.DataFrame): DataFrame with importance values
        title (str): Plot title
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
    """
    normalized_df = importance_df.apply(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x,
        axis=0
    )
    
    plt.figure(figsize=figsize)
    sns.heatmap(normalized_df, annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
    plt.title(title)
    plt.xlabel("Method / Model")
    plt.ylabel("Feature")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_shap_waterfall(shap_explanation, title, save_path=None):
    """
    Plot a SHAP waterfall plot.
    
    Args:
        shap_explanation: SHAP Explanation object
        title (str): Plot title
        save_path (str, optional): Path to save the figure
    """
    import shap
    plt.figure()
    shap.plots.waterfall(shap_explanation, show=False)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_lime_explanation(lime_explanation, title, save_path=None):
    """
    Plot a LIME explanation.
    
    Args:
        lime_explanation: LIME explanation object
        title (str): Plot title
        save_path (str, optional): Path to save the figure
    """
    fig = lime_explanation.as_pyplot_figure()
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

