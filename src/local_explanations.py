"""
Local explanation methods for XAI analysis.

This module implements local explanation methods including:
- SHAP local explanations
- LIME explanations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
try:
    from . import visualization
except ImportError:
    import visualization


def extract_lime_features(lime_exp_list, feature_names):
    """
    Extract feature weights from LIME explanation list.
    
    Args:
        lime_exp_list (list): List of tuples from LIME explanation
        feature_names (list): List of feature names
    
    Returns:
        pd.Series: Feature weights indexed by feature names
    """
    feature_weights = {}
    for exp_str, weight in lime_exp_list:
        for fn in feature_names:
            if fn in exp_str:
                feature_weights[fn] = feature_weights.get(fn, 0) + weight
                break
    return pd.Series(feature_weights).reindex(feature_names).fillna(0)


def compute_shap_local_explanation(explainer, instance, model, feature_names, class_names):
    """
    Compute SHAP local explanation for a single instance.
    
    Args:
        explainer: SHAP explainer
        instance: Single instance to explain (numpy array)
        model: Trained model
        feature_names (list): List of feature names
        class_names (list): List of class names
    
    Returns:
        dict: Dictionary containing SHAP values and contributions
    """
    shap_values_one = explainer(instance)
    pred_class = int(model.predict(instance)[0])
    
    contrib = pd.Series(
        shap_values_one.values[0, :, pred_class],
        index=feature_names
    ).sort_values(key=lambda s: np.abs(s), ascending=False)
    
    return {
        'shap_values': shap_values_one,
        'pred_class': pred_class,
        'contributions': contrib
    }


def create_lime_explainer(X_train, feature_names, class_names, random_state=42):
    """
    Create a LIME tabular explainer.
    
    Args:
        X_train (pd.DataFrame): Training features
        feature_names (list): List of feature names
        class_names (list): List of class names
        random_state (int): Random seed
    
    Returns:
        LimeTabularExplainer: LIME explainer
    """
    X_train_np = X_train.values
    explainer = LimeTabularExplainer(
        training_data=X_train_np,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=random_state
    )
    return explainer


def compute_lime_explanation(lime_explainer, instance, model, feature_names):
    """
    Compute LIME explanation for a single instance.
    
    Args:
        lime_explainer: LIME explainer
        instance: Single instance to explain (numpy array)
        model: Trained model with predict_proba method
        feature_names (list): List of feature names
    
    Returns:
        dict: Dictionary containing LIME explanation and feature contributions
    """
    lime_exp = lime_explainer.explain_instance(
        data_row=instance[0],
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )
    
    contrib_series = extract_lime_features(lime_exp.as_list(), feature_names)
    
    return {
        'explanation': lime_exp,
        'contributions': contrib_series
    }


def select_class_explanation(shap_values_one, class_idx, feature_names):
    """
    Select SHAP explanation for a specific class.
    
    Args:
        shap_values_one: SHAP values for one instance
        class_idx (int): Class index
        feature_names (list): List of feature names
    
    Returns:
        shap.Explanation: SHAP explanation object for the specified class
    """
    return shap.Explanation(
        values=shap_values_one.values[0, :, class_idx],
        base_values=shap_values_one.base_values[0, class_idx],
        data=shap_values_one.data[0],
        feature_names=feature_names
    )


def analyze_local_explanations(lr_clf, rf_clf, explainer_lr, explainer_rf, 
                               lime_explainer, instance_idx, X_test, y_test,
                               feature_names, class_names):
    """
    Analyze local explanations for a specific instance using both SHAP and LIME.
    
    Args:
        lr_clf: Trained Logistic Regression model
        rf_clf: Trained Random Forest model
        explainer_lr: SHAP explainer for LR
        explainer_rf: SHAP explainer for RF
        lime_explainer: LIME explainer
        instance_idx (int): Index of instance to analyze
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        feature_names (list): List of feature names
        class_names (list): List of class names
    
    Returns:
        dict: Dictionary containing all local explanation results
    """
    x_instance = X_test.iloc[instance_idx]
    x_instance_np = x_instance.values.reshape(1, -1)
    
    true_class = class_names[y_test.iloc[instance_idx]]
    pred_lr = class_names[lr_clf.predict(x_instance_np)[0]]
    pred_rf = class_names[rf_clf.predict(x_instance_np)[0]]
    
    # SHAP explanations
    shap_lr = compute_shap_local_explanation(explainer_lr, x_instance_np, lr_clf, feature_names, class_names)
    shap_rf = compute_shap_local_explanation(explainer_rf, x_instance_np, rf_clf, feature_names, class_names)
    
    # LIME explanations
    lime_lr = compute_lime_explanation(lime_explainer, x_instance_np, lr_clf, feature_names)
    lime_rf = compute_lime_explanation(lime_explainer, x_instance_np, rf_clf, feature_names)
    
    return {
        'instance_idx': instance_idx,
        'instance': x_instance,
        'true_class': true_class,
        'pred_lr': pred_lr,
        'pred_rf': pred_rf,
        'shap_lr': shap_lr,
        'shap_rf': shap_rf,
        'lime_lr': lime_lr,
        'lime_rf': lime_rf
    }


def plot_local_explanations(local_results, class_names, output_dir=None):
    """
    Generate all local explanation plots for a single instance.
    
    Args:
        local_results (dict): Results from analyze_local_explanations
        class_names (list): List of class names
        output_dir (str, optional): Directory to save plots
    """
    idx = local_results['instance_idx']
    pred_class_lr = local_results['shap_lr']['pred_class']
    pred_class_rf = local_results['shap_rf']['pred_class']
    
    # SHAP waterfall plots
    exp_lr = select_class_explanation(
        local_results['shap_lr']['shap_values'],
        pred_class_lr,
        local_results['shap_lr']['contributions'].index.tolist()
    )
    exp_rf = select_class_explanation(
        local_results['shap_rf']['shap_values'],
        pred_class_rf,
        local_results['shap_rf']['contributions'].index.tolist()
    )
    
    visualization.plot_shap_waterfall(
        exp_lr,
        f"SHAP waterfall – LR (class: {class_names[pred_class_lr]})",
        save_path=f"{output_dir}/shap_waterfall_lr_obs{idx}.png" if output_dir else None
    )
    
    visualization.plot_shap_waterfall(
        exp_rf,
        f"SHAP waterfall – RF (class: {class_names[pred_class_rf]})",
        save_path=f"{output_dir}/shap_waterfall_rf_obs{idx}.png" if output_dir else None
    )
    
    # LIME plots
    visualization.plot_lime_explanation(
        local_results['lime_lr']['explanation'],
        f"LIME – LR (pred: {local_results['pred_lr']})",
        save_path=f"{output_dir}/lime_lr_obs{idx}.png" if output_dir else None
    )
    
    visualization.plot_lime_explanation(
        local_results['lime_rf']['explanation'],
        f"LIME – RF (pred: {local_results['pred_rf']})",
        save_path=f"{output_dir}/lime_rf_obs{idx}.png" if output_dir else None
    )


def plot_local_comparison_heatmap(local_results_list, method='shap', output_dir=None):
    """
    Plot comparison heatmap for multiple instances.
    
    Args:
        local_results_list (list): List of local explanation results
        method (str): 'shap' or 'lime'
        output_dir (str, optional): Directory to save plots
    """
    import seaborn as sns
    
    comparison_data = {}
    for result in local_results_list:
        idx = result['instance_idx']
        if method == 'shap':
            comparison_data[f"LR_SHAP_Obs{idx}"] = result['shap_lr']['contributions']
            comparison_data[f"RF_SHAP_Obs{idx}"] = result['shap_rf']['contributions']
        else:  # lime
            comparison_data[f"LR_LIME_Obs{idx}"] = result['lime_lr']['contributions']
            comparison_data[f"RF_LIME_Obs{idx}"] = result['lime_rf']['contributions']
    
    comparison_df = pd.DataFrame(comparison_data).fillna(0)
    normalized_df = comparison_df.abs().apply(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x,
        axis=0
    )
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(normalized_df, annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
    plt.title(f"Normalized Local Feature Importance ({method.upper()}): Comparison of Observations")
    plt.xlabel("Model / Observation")
    plt.ylabel("Feature")
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(f"{output_dir}/local_comparison_{method}_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_combined_local_comparison(local_results, output_dir=None):
    """
    Plot combined comparison of SHAP and LIME for a single instance.
    
    Args:
        local_results (dict): Results from analyze_local_explanations
        output_dir (str, optional): Directory to save plots
    """
    import seaborn as sns
    
    idx = local_results['instance_idx']
    
    combined_df = pd.DataFrame({
        "LR_SHAP_local": local_results['shap_lr']['contributions'],
        "RF_SHAP_local": local_results['shap_rf']['contributions'],
        "LR_LIME_local": local_results['lime_lr']['contributions'],
        "RF_LIME_local": local_results['lime_rf']['contributions']
    }).fillna(0)
    
    normalized_df = combined_df.abs().apply(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x,
        axis=0
    )
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(normalized_df, annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
    plt.title(f"Normalized Local Feature Importance for Observation {idx} Across Methods and Models")
    plt.xlabel("Method / Model (local)")
    plt.ylabel("Feature")
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(f"{output_dir}/combined_local_comparison_obs{idx}.png", dpi=300, bbox_inches='tight')
    plt.close()

