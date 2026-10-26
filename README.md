# Custom Gradient Boosting Optimizer — Implementation from Scratch

A from-scratch, XGBoost-style Gradient Boosted Decision Tree (GBDT)
regressor in Python: exact greedy split-finding on second-order
gradients/Hessians, L2-regularized leaf weights, learning-rate
shrinkage, and column subsampling — no scikit-learn, no external
datasets.

## Files

| File | Purpose |
|---|---|
| `custom_gbdt.py` | The framework: `Node`, `RegressionTree`, `L2MSELoss`, `BoostingEnsemble`, `generate_synthetic_data` |
| `test_custom_gbdt.py` | `unittest` suite: loss math, split-finding logic, ensemble behavior, data generator (25 tests) |
| `analysis.py` | Runs the required hyperparameter sweep, writes `analysis_report.md` and `convergence_plot.png` |
| `analysis_report.md` | Write-up of convergence behavior, hyperparameter sensitivity, feature importance, and the second-order-vs-first-order discussion |
| `convergence_plot.png` | Training-loss-vs-iteration curves for every `max_depth` × `learning_rate` combination |

## Requirements

Python 3.8+, `numpy`, `matplotlib` (only needed to regenerate the plot).

## Running it

```bash
# run the unit tests
python3 -m unittest test_custom_gbdt -v

# regenerate the analysis report + plot
python3 analysis.py
```

## Quick usage example

```python
from custom_gbdt import BoostingEnsemble, generate_synthetic_data

X, y = generate_synthetic_data(n_samples=500, n_features=5, random_state=42)

model = BoostingEnsemble(
    n_estimators=150,
    learning_rate=0.1,   # eta
    max_depth=3,
    reg_lambda=1.0,       # L2 leaf-weight smoothing
    colsample=0.8,        # column subsampling
)
model.fit(X, y, verbose=True)

print("R^2 on training data:", model.score(X, y))
print("Feature importances:", model.feature_importances())
```

## Design notes

- **Split-finding** (`RegressionTree._find_best_split`) uses the exact
  XGBoost gain formula, vectorized with `numpy.cumsum` over each
  sorted feature column rather than a per-pair Python loop, so a full
  9-point hyperparameter sweep at 150 rounds each runs in well under
  ten seconds.
- **Leaf weights** are the closed-form optimum `w* = -G/(H+λ)`, where
  `λ` (`reg_lambda`) is the L2 regularization / "leaf-weight
  smoothing" term called for in the project spec.
- See `analysis_report.md` §3 for why this project's MSE loss makes
  the Hessian a constant (`h_i = 1`), and what that does — and does
  not — imply about second-order vs. first-order convergence speed.
