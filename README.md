# XAI Analysis: Consistency of Results

Analysis of explainable AI methods on the Iris dataset. This project compares different explanation techniques (coefficients, feature importance, permutation importance, SHAP, LIME) across Logistic Regression and Random Forest classifiers to assess consistency of feature importance rankings.

## Project Structure

```
XAI_Analisys/
├── src/
│   ├── data_loading.py          # Data loading and preprocessing
│   ├── models.py                # Model training and evaluation
│   ├── global_explanations.py   # Global explanation methods
│   ├── local_explanations.py    # Local explanation methods
│   ├── visualization.py       # Plotting utilities
│   └── main.py                  # Main execution script
├── outputs/
│   ├── figures/                 # Generated plots
│   ├── models/                  # Saved model artifacts
│   └── reports/                 # Analysis reports
├── notebooks/                   # Original notebook (reference)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd XAI_Analisys
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete analysis:

```bash
python src/main.py
```

This will:
- Load and prepare the Iris dataset
- Train Logistic Regression and Random Forest models
- Generate global explanations (coefficients, feature importance, permutation importance, SHAP)
- Generate local explanations (SHAP, LIME) for selected instances
- Save all plots to `outputs/figures/`

## Methods Analyzed

### Global Explanations
- **Model Coefficients**: Logistic Regression coefficients (after standardization)
- **Feature Importance**: Random Forest built-in feature importances
- **Permutation Importance**: Accuracy drop after feature shuffling
- **SHAP Global**: Mean absolute SHAP values across all instances

### Local Explanations
- **SHAP Local**: Feature contributions for individual predictions
- **LIME**: Local interpretable model-agnostic explanations

## Results

The analysis compares feature importance rankings across different methods and models. Key findings include:
- Petal length and petal width consistently rank as most important features
- Sepal features (especially sepal width) have lower importance
- Different explanation methods show consistent patterns despite different underlying mechanisms

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- shap
- lime
- eli5

## License

See LICENSE file for details.
