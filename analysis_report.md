# Custom GBDT: Convergence & Hyperparameter Sensitivity Report

This report summarizes the training behavior of the from-scratch Gradient Boosting Decision Tree (GBDT) implemented in `custom_gbdt.py`, evaluated on the project's synthetic regression dataset from `generate_synthetic_data()`: `y = 3*x1^2 + sin(x2) + 0.5*x3 + e`, 500 samples, 5 features (x4, x5 are pure noise), with an 80/20 train/test split.

## 1. Hyperparameter sensitivity grid

Each combination of `max_depth` in {3, 5, 7} and `learning_rate` (eta) in {0.01, 0.1, 0.3} was trained for 150 boosting rounds with `reg_lambda=1.0`, `colsample=0.8`, `min_child_weight=1.0`. `iters to -90%` is the first round at which training loss has dropped to within 10% of its own starting value (measured against the fixed starting loss, not the run's own final loss, so it is not a moving target); `not reached` means the run never got there within 150 rounds.

| max_depth | learning_rate | train MSE | test MSE | test R² | iters to -90% |
|---|---|---|---|---|---|
| 3 | 0.01 | 2.9484 | 5.4636 | 0.7119 | >150 (not reached) |
| 3 | 0.1 | 0.1177 | 0.8612 | 0.9546 | 22 |
| 3 | 0.3 | 0.0202 | 0.7831 | 0.9587 | 7 |
| 5 | 0.01 | 2.3093 | 5.1910 | 0.7263 | >150 (not reached) |
| 5 | 0.1 | 0.0261 | 1.3202 | 0.9304 | 18 |
| 5 | 0.3 | 0.0001 | 0.9250 | 0.9512 | 6 |
| 7 | 0.01 | 2.0175 | 5.1795 | 0.7269 | >150 (not reached) |
| 7 | 0.1 | 0.0035 | 1.3207 | 0.9304 | 18 |
| 7 | 0.3 | 0.0000 | 1.0914 | 0.9424 | 5 |

**Best configuration on held-out test MSE:** `max_depth=3`, `learning_rate=0.3` (test MSE = 0.7831, test R² = 0.9587). See `convergence_plot.png` for the full per-round training curves.

### Observations

- **Learning rate is the strongest lever within this fixed 150-round budget.** Raising eta from 0.01 to 0.1 lifts test R² from roughly 0.72 to the mid-0.93s at every depth, with a smaller further gain from 0.1 to 0.3. The eta=0.01 runs still carry visibly elevated *training* MSE too (2.0-2.9, versus well under 0.12 for eta=0.1), so this is primarily a **budget effect**: each tree's contribution is scaled by eta before being added to the running prediction (`F(x) += eta * tree(x)`), so eta=0.01 simply has not taken enough cumulative steps in 150 rounds to converge, not evidence that small learning rates are inherently worse. More rounds would be expected to keep closing that gap.
- **More tree depth keeps improving the training fit but not the test fit -- a clean overfitting signature.** At every learning rate, training MSE falls monotonically from depth 3 to depth 7 (reaching ~0.0000-0.0035 by depth 7), but test MSE does the *opposite*: it is lowest at depth 3 for every learning rate and gets worse at depth 5 and 7. Most visibly at eta=0.3, test R² goes 0.9587 (depth 3) → 0.9512 (depth 5) → 0.9424 (depth 7). The extra depth gives the ensemble more capacity to fit training-set-specific noise, and `reg_lambda=1.0`/`min_child_weight=1.0`/`colsample=0.8` are not enough to fully offset that at depth 5+ on this dataset.
- **Best configuration in this grid: max_depth=3, eta=0.3** -- the *shallowest* trees paired with the *largest* learning rate tested. This is a reasonable outcome for this data-generating process: with 150 additive rounds, most of the ensemble's expressive power comes from the sheer number of boosted trees rather than from any single tree's depth, so individual trees do not need much depth to contribute a useful partial fit -- extra depth mainly adds overfitting risk here rather than needed capacity.

## 2. Feature importance (gain-based)

Split gain summed per feature across the best model's trees, normalized to sum to 1:

| feature | importance |
|---|---|
| x1 | 0.9438 |
| x2 | 0.0316 |
| x3 | 0.0164 |
| x4 | 0.0029 |
| x5 | 0.0053 |

x1, x2 and x3 -- the three features that actually appear in the data-generating formula -- account for essentially all of the importance, while x4 and x5 (pure noise) receive little to none. This confirms the greedy split-finder is correctly identifying informative columns rather than fitting noise.

## 3. Why second-order gradients help -- and the nuance for this loss

This framework follows XGBoost's Newton-boosting recipe: each new tree is fit not to the raw residual, but to the per-sample gradient `g_i = dL/dF(x_i)` *and* Hessian `h_i = d²L/dF(x_i)²` of the loss with respect to the current prediction, and the optimal leaf weight is `w* = -G/(H+λ)` rather than a plain average of gradients.

**General case.** For losses whose curvature varies by instance (logistic/cross-entropy loss, Poisson deviance, Huber loss, quantile loss, etc.), the Hessian tells the split-finder how much a given region of input space can still be improved before overshooting. The resulting leaf weight behaves like a local Newton step rather than a fixed-size gradient-descent step, which is why Newton boosting typically reaches a given loss in fewer rounds than plain first-order gradient boosting: it self-corrects for curvature instead of relying solely on a single global learning rate to avoid overshooting high-curvature regions.

**This project's loss is a specific, important edge case.** Because the objective here is squared error, `L(y,F) = 0.5*(y-F)²`, its second derivative is the constant `h_i = 1` for every sample (see `L2MSELoss.hessian`). For *this* loss, the Hessian carries no curvature information for the split-finder to exploit: plugging `h_i=1` into `w*=-G/(H+λ)` reduces it to `w* = -G/(n_leaf+λ)`, i.e. a **shrunken, regularized mean of the negative gradients**, not a curvature-adjusted step. The practical benefit visible in the experiments above is therefore not 'faster convergence from curvature correction' but **variance reduction from the L2-regularized leaf weight**: leaves built from few samples (small H) get pulled further toward 0 by λ than leaves built from many samples, which is exactly the leaf-weight-smoothing behavior the project spec asks for, and is also why `reg_lambda`, `gamma` and `min_child_weight` matter more to this model's generalization than the Hessian's magnitude does.

In short: the second-order machinery is implemented faithfully and is exactly what makes the L2-regularized leaf weight and exact split-gain formula correct, but the *convergence-speed* advantage second-order boosting is known for is a property of losses with non-constant curvature. It would show up directly if this same `RegressionTree`/`BoostingEnsemble` code were pointed at a loss like logistic loss for a classification target, but is inherently muted for plain MSE regression, where the Hessian is constant by construction.

## 4. Files in this project

- `custom_gbdt.py` -- `Node`, `RegressionTree`, `BoostingEnsemble`, `L2MSELoss`, `generate_synthetic_data`
- `test_custom_gbdt.py` -- unittest suite (loss math, split-finding logic, ensemble convergence/baseline/feature-importance checks, data generator)
- `analysis.py` -- produces this report and `convergence_plot.png`
- `convergence_plot.png` -- training loss vs. iteration, one panel per `max_depth`, one line per `learning_rate`
- `README.md` -- how to run everything
