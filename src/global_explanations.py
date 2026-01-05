"""
Global explanation methods for XAI analysis.

This module implements various global explanation methods including:
- Model coefficients/feature importances
- Permutation importance
- SHAP global explanations
"""

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
try:
    from . import visualization
except ImportError:
    import visualization


def compute_lr_coefficients(lr_clf, feature_names, class_names):
    """
    Extract and analyze Logistic Regression coefficients.
    
    Args:
        lr_clf: Trained Logistic Regression pipeline
        feature_names (list): List of feature names
        class_names (list): List of class names
    
    Returns:
        tuple: (coef_df, lr_global_strength)
            - coef_df: DataFrame with coefficients per class
            - lr_global_strength: Series with mean absolute coefficients
    """
    lr_model = lr_clf.named_steps["model"]
    coef = lr_model.coef_
    
    coef_df = pd.DataFrame(coef, columns=feature_names, index=class_names)
    lr_global_strength = coef_df.abs().mean(axis=0).sort_values(ascending=False)
    
    return coef_df, lr_global_strength


def compute_rf_feature_importance(rf_clf, feature_names):
    """
    Extract Random Forest feature importances.
    
    Args:
        rf_clf: Trained Random Forest model
        feature_names (list): List of feature names
    
    Returns:
        pd.Series: Feature importances sorted in descending order
    """
    rf_imp = pd.Series(
        rf_clf.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)
    
    return rf_imp


def compute_permutation_importance(model, X_test, y_test, feature_names, n_repeats=30, random_state=42):
    """
    Compute permutation importance for a model.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        feature_names (list): List of feature names
        n_repeats (int): Number of times to permute each feature
        random_state (int): Random seed
    
    Returns:
        pd.Series: Permutation importance values sorted in descending order
    """
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="accuracy"
    )
    return pd.Series(
        result.importances_mean,
        index=feature_names
    ).sort_values(ascending=False)


def compute_shap_global_importance(shap_values, feature_names):
    """
    Compute global SHAP importance from SHAP values.
    
    Args:
        shap_values: SHAP values object
        feature_names (list): List of feature names
    
    Returns:
        pd.Series: Global SHAP importance sorted in descending order
    """
    vals = shap_values.values
    shap_imp = pd.Series(
        np.mean(np.abs(vals), axis=(0, 2)),
        index=feature_names
    ).sort_values(ascending=False)
    
    return shap_imp


def create_shap_explainers(lr_clf, rf_clf, X_train, feature_names):
    """
    Create SHAP explainers for both models.
    
    Args:
        lr_clf: Trained Logistic Regression model
        rf_clf: Trained Random Forest model
        X_train (pd.DataFrame): Training features
        feature_names (list): List of feature names
    
    Returns:
        tuple: (explainer_lr, explainer_rf)
    """
    X_train_np = X_train.values
    
    explainer_lr = shap.Explainer(
        lr_clf.predict_proba,
        X_train_np,
        feature_names=feature_names
    )
    
    explainer_rf = shap.Explainer(
        rf_clf.predict_proba,
        X_train_np,
        feature_names=feature_names
    )
    
    return explainer_lr, explainer_rf


def compute_all_global_explanations(lr_clf, rf_clf, X_train, X_test, y_test, feature_names, class_names):
    """
    Compute all global explanation methods and return results.
    
    Args:
        lr_clf: Trained Logistic Regression model
        rf_clf: Trained Random Forest model
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        feature_names (list): List of feature names
        class_names (list): List of class names
    
    Returns:
        dict: Dictionary containing all global explanation results
    """
    # LR coefficients
    coef_df, lr_global_strength = compute_lr_coefficients(lr_clf, feature_names, class_names)
    
    # RF feature importance
    rf_imp = compute_rf_feature_importance(rf_clf, feature_names)
    
    # Permutation importance
    perm_lr = compute_permutation_importance(lr_clf, X_test, y_test, feature_names)
    perm_rf = compute_permutation_importance(rf_clf, X_test, y_test, feature_names)
    
    # SHAP explainers and values
    explainer_lr, explainer_rf = create_shap_explainers(lr_clf, rf_clf, X_train, feature_names)
    X_test_np = X_test.values
    shap_values_lr = explainer_lr(X_test_np)
    shap_values_rf = explainer_rf(X_test_np)
    
    # SHAP global importance
    shap_imp_lr = compute_shap_global_importance(shap_values_lr, feature_names)
    shap_imp_rf = compute_shap_global_importance(shap_values_rf, feature_names)
    
    return {
        'coef_df': coef_df,
        'lr_global_strength': lr_global_strength,
        'rf_imp': rf_imp,
        'perm_lr': perm_lr,
        'perm_rf': perm_rf,
        'explainer_lr': explainer_lr,
        'explainer_rf': explainer_rf,
        'shap_values_lr': shap_values_lr,
        'shap_values_rf': shap_values_rf,
        'shap_imp_lr': shap_imp_lr,
        'shap_imp_rf': shap_imp_rf
    }


def create_global_comparison_summary(lr_global_strength, perm_lr, shap_imp_lr, rf_imp, perm_rf, shap_imp_rf, k=4):
    """
    Create a summary DataFrame comparing top features across methods.
    
    Args:
        lr_global_strength: LR coefficient importance
        perm_lr: LR permutation importance
        shap_imp_lr: LR SHAP importance
        rf_imp: RF feature importance
        perm_rf: RF permutation importance
        shap_imp_rf: RF SHAP importance
        k (int): Number of top features to include
    
    Returns:
        pd.DataFrame: Summary comparison
    """
    def top_k(series, k_val):
        return list(series.index[:k_val])
    
    summary_global = pd.DataFrame({
        "LR_coef_top": [', '.join(top_k(lr_global_strength, k))],
        "LR_perm_top": [', '.join(top_k(perm_lr, k))],
        "LR_shap_top": [', '.join(top_k(shap_imp_lr, k))],
        "RF_builtin_top": [', '.join(top_k(rf_imp, k))],
        "RF_perm_top": [', '.join(top_k(perm_rf, k))],
        "RF_shap_top": [', '.join(top_k(shap_imp_rf, k))]
    })
    
    return summary_global.T


def plot_global_explanations(global_results, output_dir=None):
    """
    Generate all global explanation plots.
    
    Args:
        global_results (dict): Results from compute_all_global_explanations
        output_dir (str, optional): Directory to save plots
    """
    lr_global_strength = global_results['lr_global_strength']
    rf_imp = global_results['rf_imp']
    perm_lr = global_results['perm_lr']
    perm_rf = global_results['perm_rf']
    shap_imp_lr = global_results['shap_imp_lr']
    shap_imp_rf = global_results['shap_imp_rf']
    
    # Plot LR coefficients
    visualization.plot_importance_bar(
        lr_global_strength,
        "LR: Mean Absolute Coefficient Values Across Classes",
        "mean |coef|",
        save_path=f"{output_dir}/lr_coefficients.png" if output_dir else None
    )
    
    # Plot RF feature importance
    visualization.plot_importance_bar(
        rf_imp,
        "RF: Feature Importances (Built-in)",
        "importance",
        save_path=f"{output_dir}/rf_feature_importance.png" if output_dir else None
    )
    
    # Plot permutation importance
    visualization.plot_permutation_importance_comparison(
        perm_lr, perm_rf,
        save_path=f"{output_dir}/permutation_importance.png" if output_dir else None
    )
    
    # Plot SHAP global importance
    visualization.plot_shap_global_comparison(
        shap_imp_lr, shap_imp_rf,
        save_path=f"{output_dir}/shap_global.png" if output_dir else None
    )
    
    # Plot normalized heatmap
    combined_importance_df = pd.DataFrame({
        "LR_Coef": lr_global_strength,
        "LR_Permutation": perm_lr,
        "LR_SHAP": shap_imp_lr,
        "RF_Builtin": rf_imp,
        "RF_Permutation": perm_rf,
        "RF_SHAP": shap_imp_rf,
    }).fillna(0)
    
    visualization.plot_normalized_heatmap(
        combined_importance_df,
        "Normalized Feature Importance Across Different Methods and Models",
        save_path=f"{output_dir}/global_importance_heatmap.png" if output_dir else None
    )

