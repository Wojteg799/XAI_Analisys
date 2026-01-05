"""
Main execution script for XAI analysis.

This script orchestrates the complete workflow:
1. Load and prepare data
2. Train models
3. Generate global explanations
4. Generate local explanations
5. Create visualizations
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src.data_loading import load_iris_data, split_data
from src.models import train_logistic_regression, train_random_forest, evaluate_models, find_misclassified_instance
from src.global_explanations import compute_all_global_explanations, plot_global_explanations, create_global_comparison_summary
from src.local_explanations import create_lime_explainer, analyze_local_explanations, plot_local_explanations, plot_local_comparison_heatmap, plot_combined_local_comparison
from src.visualization import plot_feature_distributions


def setup_output_directories():
    """Create output directories if they don't exist."""
    output_dirs = [
        'outputs',
        'outputs/figures',
        'outputs/models',
        'outputs/reports'
    ]
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
    return 'outputs/figures'


def main():
    """Main execution function."""
    print("=" * 60)
    print("XAI Analysis: Consistency of Results")
    print("=" * 60)
    
    # Setup output directories
    output_dir = setup_output_directories()
    
    # 1. Load data
    print("\n[1/5] Loading Iris dataset...")
    X, y, feature_names, class_names, df = load_iris_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution:\n{df['class_name'].value_counts()}")
    
    # Plot feature distributions
    plot_feature_distributions(X, save_path=f"{output_dir}/feature_distributions.png")
    
    # Split data
    print("\n[2/5] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 2. Train models
    print("\n[3/5] Training models...")
    print("Training Logistic Regression...")
    lr_clf = train_logistic_regression(X_train, y_train, random_state=42)
    
    print("Training Random Forest...")
    rf_clf = train_random_forest(X_train, y_train, n_estimators=300, random_state=42)
    
    # Evaluate models
    print("\nModel Evaluation:")
    lr_accuracy, rf_accuracy, lr_pred, rf_pred = evaluate_models(
        lr_clf, rf_clf, X_test, y_test, class_names
    )
    
    # 3. Global explanations
    print("\n[4/5] Computing global explanations...")
    global_results = compute_all_global_explanations(
        lr_clf, rf_clf, X_train, X_test, y_test,
        feature_names, class_names
    )
    
    print("Generating global explanation plots...")
    plot_global_explanations(global_results, output_dir=output_dir)
    
    # Print summary
    summary = create_global_comparison_summary(
        global_results['lr_global_strength'],
        global_results['perm_lr'],
        global_results['shap_imp_lr'],
        global_results['rf_imp'],
        global_results['perm_rf'],
        global_results['shap_imp_rf']
    )
    print("\nGlobal Feature Importance Summary:")
    print(summary)
    
    # 4. Local explanations
    print("\n[5/5] Computing local explanations...")
    
    # Create LIME explainer
    lime_explainer = create_lime_explainer(X_train, feature_names, class_names, random_state=42)
    
    # Analyze observation 0 (correctly classified by both)
    print("\nAnalyzing observation 0 (correctly classified)...")
    local_results_0 = analyze_local_explanations(
        lr_clf, rf_clf,
        global_results['explainer_lr'],
        global_results['explainer_rf'],
        lime_explainer,
        0, X_test, y_test,
        feature_names, class_names
    )
    
    print(f"True class: {local_results_0['true_class']}")
    print(f"LR prediction: {local_results_0['pred_lr']}")
    print(f"RF prediction: {local_results_0['pred_rf']}")
    
    plot_local_explanations(local_results_0, class_names, output_dir=output_dir)
    plot_combined_local_comparison(local_results_0, output_dir=output_dir)
    
    # Find and analyze misclassified instance
    misclassified_idx = find_misclassified_instance(lr_clf, rf_clf, X_test, y_test, start_idx=1)
    
    if misclassified_idx is not None:
        print(f"\nAnalyzing observation {misclassified_idx} (misclassified)...")
        local_results_mis = analyze_local_explanations(
            lr_clf, rf_clf,
            global_results['explainer_lr'],
            global_results['explainer_rf'],
            lime_explainer,
            misclassified_idx, X_test, y_test,
            feature_names, class_names
        )
        
        print(f"True class: {local_results_mis['true_class']}")
        print(f"LR prediction: {local_results_mis['pred_lr']}")
        print(f"RF prediction: {local_results_mis['pred_rf']}")
        
        plot_local_explanations(local_results_mis, class_names, output_dir=output_dir)
        plot_combined_local_comparison(local_results_mis, output_dir=output_dir)
        
        # Comparison heatmaps
        print("\nGenerating comparison heatmaps...")
        plot_local_comparison_heatmap(
            [local_results_0, local_results_mis],
            method='shap',
            output_dir=output_dir
        )
        plot_local_comparison_heatmap(
            [local_results_0, local_results_mis],
            method='lime',
            output_dir=output_dir
        )
    else:
        print("\nNo misclassified instances found (excluding index 0).")
    
    print("\n" + "=" * 60)
    print("Analysis complete! All outputs saved to 'outputs/' directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()

