# Thyroid Disease Detection: Comparative Analysis of Classification Models

A didactic "Data Science Lab" for building robust diagnostic tools for thyroid anomalies using the UCI Thyroid Disease dataset.

## Learning Objectives

1. **Imbalanced Data**: Handling ~90% "negative" class vs. critical disease classes
2. **Informative Missingness**: Distinguishing "missing by design" (TBG) vs. "missing by chance" (T3)
3. **Pipeline Engineering**: Using scikit-learn pipelines for preprocessing
4. **Model Comparison**: Baselines → Gradient Boosting → Neural Networks

## Project Structure

| Notebook                    | Description                                   |
| --------------------------- | --------------------------------------------- |
| `01_data_preparation.ipynb` | Dataset loading, EDA, train/test split        |
| `02_baseline_models.ipynb`  | Logistic Regression & Random Forest baselines |
| `03_advanced_models.ipynb`  | XGBoost & CatBoost with hyperparameter tuning |
| `04_neural_network.ipynb`   | PyTorch Feed-Forward Network                  |
| `05_evaluation.ipynb`       | Final comparison on hold-out test set         |

## Dataset

**Source:** [Kaggle - Thyroid Disease Data](https://www.kaggle.com/datasets/emmanuelfwerr/thyroid-disease-data)

**Target:** Simplified to 3 classes: `negative`, `hyperthyroid`, `hypothyroid`

## Preprocessing Strategies

### Simple Imputation (Logistic Regression)

- Global median imputation for numerics
- Mode imputation for categoricals
- StandardScaler for gradient-based optimization

### Native NaN Handling (Random Forest, XGBoost)

- Pass NaN directly to tree models (sklearn >= 1.8)
- Drop `_measured` flags (redundant for trees)

### Zero Imputation with Flags (Neural Networks)

- Fill missing with 0.0
- Keep ALL `_measured` flags (network learns missingness patterns)

## Evaluation Metric

**Thyroid Mean Recall**: Average recall of hyperthyroid and hypothyroid classes

$$\text{Score} = \frac{\text{Recall}_{hyper} + \text{Recall}_{hypo}}{2}$$

## Requirements

- Python >= 3.13
- scikit-learn >= 1.6
- xgboost, torch, pandas, seaborn
- catboost (optional, install separately if needed)

Install dependencies:

```bash
uv sync
```

## Development Notes

### Module Imports

All notebooks are designed to be executed from the project root directory. This ensures that local modules within the `src/` package are correctly discovered without requiring explicit modifications to `sys.path`. This approach maintains code cleanliness and consistency across the entire lab series.

## Future Work

- KNNImputer for T3 conditional imputation (T3 is biologically correlated with T4 and TSH)
- Class weighting or SMOTE for imbalanced handling
- Model calibration for probability outputs
