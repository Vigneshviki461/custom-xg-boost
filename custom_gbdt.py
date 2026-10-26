"""
custom_gbdt.py

A from-scratch implementation of a Gradient Boosted Decision Tree (GBDT)
regressor built on second-order (Newton) boosting, in the style of
XGBoost: each tree is fit to the per-sample gradient and Hessian of the
loss with respect to the current ensemble prediction, and leaf weights
are solved in closed form with an L2 regularization term.

Classes
-------
Node             : a single node (internal or leaf) of a RegressionTree
RegressionTree   : one regression tree, grown by exact greedy, gradient/
                   Hessian-weighted split-finding
L2MSELoss        : squared-error loss and its gradient/Hessian
BoostingEnsemble : orchestrates sequential tree fitting with learning-rate
                   shrinkage (eta) and column (feature) subsampling

Functions
---------
generate_synthetic_data : builds the project's synthetic regression
                           dataset, y = 3*x1^2 + sin(x2) + 0.5*x3 + noise
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_samples: int = 500,
    n_features: int = 5,
    noise_std: float = 0.5,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic non-linear regression dataset.

    The target is

        y = 3 * x1**2 + sin(x2) + 0.5 * x3 + e,    e ~ N(0, noise_std**2)

    x1 .. x_{n_features} are drawn i.i.d. from a standard normal
    distribution. Only x1, x2 and x3 carry signal; any remaining columns
    (x4, x5, ...) are pure noise features included on purpose, so the
    tree-splitting logic has to discover which columns actually matter
    instead of the test suite simply trusting that every feature is used.

    Parameters
    ----------
    n_samples : number of rows to generate.
    n_features : number of feature columns, must be >= 3.
    noise_std : standard deviation of the additive Gaussian noise term.
    random_state : seed for reproducibility (None for non-deterministic).

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    """
    if n_features < 3:
        raise ValueError("n_features must be >= 3 for this data-generating process")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    rng = np.random.default_rng(random_state)
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_samples)

    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    y = 3.0 * x1 ** 2 + np.sin(x2) + 0.5 * x3 + noise
    return X, y


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class L2MSELoss:
    """Squared-error loss, L(y, F) = 0.5 * (y - F)**2.

    Supplies the first derivative (gradient) and second derivative
    (Hessian) of the loss with respect to the current prediction F, which
    BoostingEnsemble uses to fit each new tree (Newton / second-order
    boosting, as in XGBoost). ``reg_lambda`` is the L2 penalty applied to
    leaf weights at tree-fitting time.

    Note: for squared error specifically, the Hessian is the constant 1
    for every sample (see ``hessian`` below) -- see the project's
    analysis report for what this does and does not imply about
    second-order vs. first-order boosting for this particular loss.
    """

    def __init__(self, reg_lambda: float = 1.0):
        self.reg_lambda = reg_lambda

    @staticmethod
    def loss(y: np.ndarray, F: np.ndarray) -> float:
        """Mean loss over all samples (for monitoring/plotting only)."""
        return float(np.mean(0.5 * (y - F) ** 2))

    @staticmethod
    def gradient(y: np.ndarray, F: np.ndarray) -> np.ndarray:
        """g_i = dL/dF = F - y."""
        return F - y

    @staticmethod
    def hessian(y: np.ndarray, F: np.ndarray) -> np.ndarray:
        """h_i = d^2L/dF^2 = 1 for every sample."""
        return np.ones_like(y, dtype=float)


# ---------------------------------------------------------------------------
# Regression tree
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """A single node of a RegressionTree.

    A node is a leaf unless ``feature_index``/``threshold`` are set; the
    recursive tree structure is just ``left``/``right`` Node references.
    """

    is_leaf: bool = True
    weight: float = 0.0
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional[Node] = None
    right: Optional[Node] = None
    gain: float = 0.0
    n_samples: int = 0


class RegressionTree:
    """A single regression tree fit on (gradient, Hessian) pairs.

    Uses the exact greedy, Hessian-weighted split-finding rule from
    XGBoost. For a candidate split into left (L) / right (R) children:

        Gain = 0.5 * [ GL^2/(HL+lambda) + GR^2/(HR+lambda)
                       - (GL+GR)^2/(HL+HR+lambda) ]  -  gamma

    and the optimal weight of a leaf with gradient sum G and Hessian sum
    H is

        w* = -G / (H + lambda)

    Parameters
    ----------
    max_depth : maximum tree depth (root is depth 0).
    reg_lambda : L2 regularization on leaf weights -- this is the
        "leaf-weight smoothing" that shrinks leaves trained on little
        data (small H) toward 0, which is what keeps the ensemble from
        overfitting to individual noisy points.
    gamma : minimum gain required to accept a split (complexity penalty
        per additional leaf).
    min_child_weight : minimum Hessian sum allowed in a child node; for
        this project's MSE loss (h_i = 1 per sample) this is effectively
        a minimum-samples-per-leaf constraint.
    feature_indices : optional subset of column indices this tree is
        restricted to splitting on (used by BoostingEnsemble to
        implement column/feature subsampling).
    """

    def __init__(
        self,
        max_depth: int = 3,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        feature_indices: Optional[np.ndarray] = None,
    ):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.feature_indices = feature_indices
        self.root: Optional[Node] = None

    # -- core XGBoost-style math -----------------------------------------

    def _leaf_weight(self, G: float, H: float) -> float:
        return -G / (H + self.reg_lambda)

    # -- fitting -----------------------------------------------------------

    def fit(self, X: np.ndarray, g: np.ndarray, h: np.ndarray) -> RegressionTree:
        """Grow the tree greedily from the given gradients/Hessians."""
        X = np.asarray(X, dtype=float)
        g = np.asarray(g, dtype=float)
        h = np.asarray(h, dtype=float)
        indices = np.arange(X.shape[0])
        self.root = self._build(X, g, h, indices, depth=0)
        return self

    def _build(self, X, g, h, indices, depth) -> Node:
        G, H = g[indices].sum(), h[indices].sum()
        node = Node(is_leaf=True, weight=self._leaf_weight(G, H), n_samples=len(indices))

        if depth >= self.max_depth or len(indices) < 2:
            return node

        best = self._find_best_split(X, g, h, indices)
        if best is None:
            return node

        feat_idx, threshold, gain, left_idx, right_idx = best

        node.is_leaf = False
        node.feature_index = int(feat_idx)
        node.threshold = float(threshold)
        node.gain = float(gain)
        node.left = self._build(X, g, h, left_idx, depth + 1)
        node.right = self._build(X, g, h, right_idx, depth + 1)
        return node

    def _find_best_split(self, X, g, h, indices):
        """Exact greedy split search, vectorized with cumulative sums.

        For every candidate feature, sort the node's samples by that
        feature's value, then use a cumulative sum of gradients/Hessians
        to evaluate the exact XGBoost split-gain formula at every
        possible split point in one vectorized pass (no inner Python
        loop over sample pairs), and keep the best (feature, threshold).
        """
        best_gain = 0.0
        best = None

        candidate_features = (
            self.feature_indices if self.feature_indices is not None
            else range(X.shape[1])
        )

        G_total = g[indices].sum()
        H_total = h[indices].sum()

        for feat_idx in candidate_features:
            values = X[indices, feat_idx]
            order = np.argsort(values, kind="mergesort")
            sorted_idx = indices[order]
            sorted_vals = values[order]
            sorted_g = g[sorted_idx]
            sorted_h = h[sorted_idx]

            # G_left[i]/H_left[i]: stats of the left child if we split
            # right after sorted position i (i = 0 .. n-2)
            G_left = np.cumsum(sorted_g)[:-1]
            H_left = np.cumsum(sorted_h)[:-1]
            G_right = G_total - G_left
            H_right = H_total - H_left

            # only split between rows with genuinely different feature
            # values, and only where both children keep enough Hessian
            # mass (min_child_weight)
            valid = sorted_vals[:-1] != sorted_vals[1:]
            valid &= H_left >= self.min_child_weight
            valid &= H_right >= self.min_child_weight

            if not np.any(valid):
                continue

            gains = 0.5 * (
                (G_left ** 2) / (H_left + self.reg_lambda)
                + (G_right ** 2) / (H_right + self.reg_lambda)
                - (G_total ** 2) / (H_total + self.reg_lambda)
            ) - self.gamma
            gains = np.where(valid, gains, -np.inf)

            pos = int(np.argmax(gains))
            local_best_gain = gains[pos]

            if local_best_gain > best_gain:
                threshold = (sorted_vals[pos] + sorted_vals[pos + 1]) / 2.0
                left_idx = sorted_idx[: pos + 1]
                right_idx = sorted_idx[pos + 1:]
                best_gain = local_best_gain
                best = (feat_idx, threshold, local_best_gain, left_idx, right_idx)

        return best

    # -- prediction --------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_row(row, self.root) for row in X])

    def _predict_row(self, row: np.ndarray, node: Node) -> float:
        if node.is_leaf:
            return node.weight
        branch = node.left if row[node.feature_index] <= node.threshold else node.right
        return self._predict_row(row, branch)


# ---------------------------------------------------------------------------
# Boosting ensemble
# ---------------------------------------------------------------------------

class BoostingEnsemble:
    """Orchestrates sequential RegressionTree fitting (Newton boosting).

    At each round, gradients/Hessians of the loss are computed against
    the ensemble's current prediction F, a new tree is fit to them (over
    a random subset of columns if ``colsample`` < 1), and the tree's
    predictions are added to F after being scaled by ``learning_rate``
    (eta) -- this shrinkage is what keeps any single tree from
    dominating the ensemble.

    Parameters
    ----------
    n_estimators : number of boosting rounds (trees).
    learning_rate : eta; shrinks each tree's contribution to F.
    max_depth, reg_lambda, gamma, min_child_weight : passed through to
        every RegressionTree (see RegressionTree's docstring).
    colsample : fraction of features randomly sampled for each tree,
        in (0, 1]. 1.0 disables column subsampling.
    random_state : seed controlling column subsampling.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        colsample: float = 1.0,
        random_state: Optional[int] = 42,
    ):
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0, 1]")
        if not (0.0 < colsample <= 1.0):
            raise ValueError("colsample must be in (0, 1]")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.colsample = colsample
        self.random_state = random_state

        self.loss = L2MSELoss(reg_lambda=reg_lambda)
        self.trees: List[RegressionTree] = []
        self.base_prediction: float = 0.0
        self.train_loss_history: List[float] = []
        self.n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> BoostingEnsemble:
        """Fit the ensemble, recording training loss after every round."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        n_sub = max(1, int(round(self.colsample * n_features)))

        self.base_prediction = float(np.mean(y))
        F = np.full(n_samples, self.base_prediction, dtype=float)

        self.trees = []
        self.train_loss_history = [self.loss.loss(y, F)]

        for m in range(self.n_estimators):
            g = self.loss.gradient(y, F)
            h = self.loss.hessian(y, F)

            feature_indices = (
                rng.choice(n_features, size=n_sub, replace=False)
                if self.colsample < 1.0 else None
            )

            tree = RegressionTree(
                max_depth=self.max_depth,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                min_child_weight=self.min_child_weight,
                feature_indices=feature_indices,
            )
            tree.fit(X, g, h)

            F += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

            current_loss = self.loss.loss(y, F)
            self.train_loss_history.append(current_loss)

            if verbose and (m + 1) % max(1, self.n_estimators // 10) == 0:
                print(f"  iter {m + 1:>4d}/{self.n_estimators} | train MSE = {current_loss:.6f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with the full ensemble: base prediction + shrunk trees."""
        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.base_prediction, dtype=float)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R^2 coefficient of determination on the given data."""
        y = np.asarray(y, dtype=float)
        preds = self.predict(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def feature_importances(self) -> np.ndarray:
        """Gain-based feature importance, normalized to sum to 1.

        Sums the split-gain contributed by each feature across every
        tree in the ensemble -- features that never produced a
        worthwhile split (e.g. the pure-noise columns in
        generate_synthetic_data) end up with importance at or near 0.
        """
        importances = np.zeros(self.n_features_)
        for tree in self.trees:
            self._accumulate_gain(tree.root, importances)
        total = importances.sum()
        return importances / total if total > 0 else importances

    @staticmethod
    def _accumulate_gain(node: Optional[Node], importances: np.ndarray) -> None:
        if node is None or node.is_leaf:
            return
        importances[node.feature_index] += node.gain
        BoostingEnsemble._accumulate_gain(node.left, importances)
        BoostingEnsemble._accumulate_gain(node.right, importances)
