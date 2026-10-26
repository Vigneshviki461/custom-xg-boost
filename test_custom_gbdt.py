"""
test_custom_gbdt.py

Unit tests for custom_gbdt.py: validates the loss gradient/Hessian math,
the greedy split-finding logic on cases with a known correct answer, and
that the full BoostingEnsemble actually reduces training loss and beats
a trivial mean-prediction baseline.

Run with:
    python -m unittest test_custom_gbdt -v
"""

import unittest

import numpy as np

from custom_gbdt import (
    BoostingEnsemble,
    L2MSELoss,
    RegressionTree,
    generate_synthetic_data,
)


class TestL2MSELoss(unittest.TestCase):
    """Correctness of the custom loss function and its derivatives."""

    def test_gradient_matches_closed_form(self):
        y = np.array([1.0, 2.0, 3.0])
        F = np.array([0.5, 2.5, 2.0])
        expected = F - y  # d/dF [0.5*(y-F)^2] = F - y
        np.testing.assert_allclose(L2MSELoss.gradient(y, F), expected)

    def test_hessian_is_constant_one(self):
        y = np.array([1.0, 2.0, 3.0, -4.0])
        F = np.zeros_like(y)
        np.testing.assert_allclose(L2MSELoss.hessian(y, F), np.ones_like(y))

    def test_loss_decreases_as_prediction_approaches_target(self):
        y = np.array([5.0, 5.0, 5.0])
        far = np.array([0.0, 0.0, 0.0])
        near = np.array([4.9, 4.9, 4.9])
        self.assertGreater(L2MSELoss.loss(y, far), L2MSELoss.loss(y, near))

    def test_loss_zero_when_prediction_equals_target(self):
        y = np.array([1.0, -2.0, 3.5])
        self.assertAlmostEqual(L2MSELoss.loss(y, y.copy()), 0.0)

    def test_loss_matches_manual_mean_squared_error_formula(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=50)
        F = rng.normal(size=50)
        expected = np.mean(0.5 * (y - F) ** 2)
        self.assertAlmostEqual(L2MSELoss.loss(y, F), expected, places=10)


class TestRegressionTreeSplitLogic(unittest.TestCase):
    """Correctness of greedy, gradient/Hessian-weighted split-finding."""

    def setUp(self):
        # A deterministic step function: y jumps from -1 to +1 at x = 0.
        # With a zero-prediction starting point, gradients derived from
        # this target (g = F - y = -y) should force the greedy
        # split-finder to split right at the jump.
        n = 200
        x = np.linspace(-5, 5, n)
        y = np.where(x < 0, -1.0, 1.0)
        self.X = x.reshape(-1, 1)
        self.y = y
        self.g = -y          # F = 0 everywhere => g = F - y = -y
        self.h = np.ones(n)  # MSE Hessian is always 1

    def test_split_recovers_known_threshold(self):
        tree = RegressionTree(max_depth=1, reg_lambda=0.0, gamma=0.0, min_child_weight=0.0)
        tree.fit(self.X, self.g, self.h)
        self.assertFalse(tree.root.is_leaf)
        self.assertEqual(tree.root.feature_index, 0)
        self.assertAlmostEqual(tree.root.threshold, 0.0, delta=0.2)

    def test_split_gain_is_positive_for_informative_split(self):
        tree = RegressionTree(max_depth=1, reg_lambda=0.0, gamma=0.0, min_child_weight=0.0)
        tree.fit(self.X, self.g, self.h)
        self.assertGreater(tree.root.gain, 0.0)

    def test_leaf_predictions_have_correct_signs(self):
        tree = RegressionTree(max_depth=1, reg_lambda=0.0, gamma=0.0, min_child_weight=0.0)
        tree.fit(self.X, self.g, self.h)
        preds = tree.predict(self.X)
        left_mask = self.X[:, 0] < -0.2
        right_mask = self.X[:, 0] > 0.2
        # left of the jump has g=+1 (true y=-1) => w*=-G/(H+lambda) < 0
        self.assertTrue(np.all(preds[left_mask] < 0))
        # right of the jump has g=-1 (true y=+1) => w*=-G/(H+lambda) > 0
        self.assertTrue(np.all(preds[right_mask] > 0))

    def test_zero_gradient_variance_yields_no_split(self):
        n = 50
        X = np.linspace(-1, 1, n).reshape(-1, 1)
        g = np.zeros(n)
        h = np.ones(n)
        tree = RegressionTree(max_depth=3, reg_lambda=1.0, gamma=0.0, min_child_weight=0.0)
        tree.fit(X, g, h)
        self.assertTrue(tree.root.is_leaf)

    def test_max_depth_is_respected(self):
        tree = RegressionTree(max_depth=2, reg_lambda=0.1, gamma=0.0, min_child_weight=0.0)
        tree.fit(self.X, self.g, self.h)

        def depth_of(node):
            if node.is_leaf:
                return 0
            return 1 + max(depth_of(node.left), depth_of(node.right))

        self.assertLessEqual(depth_of(tree.root), 2)

    def test_min_child_weight_blocks_tiny_splits(self):
        # With min_child_weight larger than half the data, no split can
        # leave both children with enough Hessian mass.
        tree = RegressionTree(max_depth=3, reg_lambda=0.0, gamma=0.0,
                               min_child_weight=len(self.X))
        tree.fit(self.X, self.g, self.h)
        self.assertTrue(tree.root.is_leaf)

    def test_feature_subsampling_restricts_candidate_features(self):
        X = np.column_stack([self.X[:, 0], self.g])  # 2nd column trivially predictive
        tree = RegressionTree(max_depth=1, reg_lambda=0.0, gamma=0.0,
                               min_child_weight=0.0, feature_indices=np.array([0]))
        tree.fit(X, self.g, self.h)
        if not tree.root.is_leaf:
            self.assertEqual(tree.root.feature_index, 0)


class TestBoostingEnsemble(unittest.TestCase):
    """End-to-end behavior of the sequential boosting orchestrator."""

    def setUp(self):
        self.X, self.y = generate_synthetic_data(n_samples=300, n_features=5, random_state=1)

    def test_training_loss_drops_substantially(self):
        model = BoostingEnsemble(n_estimators=40, learning_rate=0.1, max_depth=3,
                                  reg_lambda=1.0, random_state=1)
        model.fit(self.X, self.y)
        history = np.array(model.train_loss_history)
        self.assertLess(history[-1], history[0] * 0.5)

    def test_training_loss_is_monotonically_non_increasing(self):
        model = BoostingEnsemble(n_estimators=40, learning_rate=0.1, max_depth=3,
                                  reg_lambda=1.0, random_state=1)
        model.fit(self.X, self.y)
        history = np.array(model.train_loss_history)
        # allow a tiny floating-point slack, but no real increases
        self.assertTrue(np.all(np.diff(history) <= 1e-9))

    def test_predictions_beat_mean_baseline(self):
        model = BoostingEnsemble(n_estimators=40, learning_rate=0.1, max_depth=3,
                                  reg_lambda=1.0, random_state=1)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)

        baseline_preds = np.full_like(self.y, np.mean(self.y))
        model_mse = np.mean((self.y - preds) ** 2)
        baseline_mse = np.mean((self.y - baseline_preds) ** 2)
        self.assertLess(model_mse, baseline_mse)

    def test_more_trees_do_not_increase_final_training_loss(self):
        shallow = BoostingEnsemble(n_estimators=5, learning_rate=0.1, max_depth=3,
                                    reg_lambda=1.0, random_state=1).fit(self.X, self.y)
        deep = BoostingEnsemble(n_estimators=40, learning_rate=0.1, max_depth=3,
                                 reg_lambda=1.0, random_state=1).fit(self.X, self.y)
        self.assertLessEqual(deep.train_loss_history[-1], shallow.train_loss_history[-1])

    def test_colsample_restricts_features_per_tree(self):
        model = BoostingEnsemble(n_estimators=10, learning_rate=0.1, max_depth=3,
                                  colsample=0.4, random_state=7)
        model.fit(self.X, self.y)
        for tree in model.trees:
            self.assertIsNotNone(tree.feature_indices)
            self.assertLessEqual(len(tree.feature_indices), self.X.shape[1])

    def test_higher_learning_rate_converges_faster_in_early_rounds(self):
        slow = BoostingEnsemble(n_estimators=10, learning_rate=0.01, max_depth=3,
                                 random_state=1).fit(self.X, self.y)
        fast = BoostingEnsemble(n_estimators=10, learning_rate=0.3, max_depth=3,
                                 random_state=1).fit(self.X, self.y)
        self.assertLess(fast.train_loss_history[-1], slow.train_loss_history[-1])

    def test_feature_importances_sum_to_one_and_favor_signal_features(self):
        model = BoostingEnsemble(n_estimators=40, learning_rate=0.1, max_depth=3,
                                  reg_lambda=1.0, random_state=1).fit(self.X, self.y)
        importances = model.feature_importances()
        self.assertEqual(len(importances), 5)
        self.assertAlmostEqual(importances.sum(), 1.0, places=6)
        # x1, x2, x3 (indices 0,1,2) carry the signal; x4, x5 are noise
        self.assertGreater(importances[:3].sum(), importances[3:].sum())

    def test_invalid_hyperparameters_raise(self):
        with self.assertRaises(ValueError):
            BoostingEnsemble(n_estimators=0)
        with self.assertRaises(ValueError):
            BoostingEnsemble(learning_rate=0.0)
        with self.assertRaises(ValueError):
            BoostingEnsemble(colsample=1.5)


class TestSyntheticDataGenerator(unittest.TestCase):
    """Correctness of the synthetic data generator."""

    def test_output_shapes(self):
        X, y = generate_synthetic_data(n_samples=500, n_features=5, random_state=0)
        self.assertEqual(X.shape, (500, 5))
        self.assertEqual(y.shape, (500,))

    def test_reproducible_with_fixed_seed(self):
        X1, y1 = generate_synthetic_data(n_samples=100, n_features=5, random_state=123)
        X2, y2 = generate_synthetic_data(n_samples=100, n_features=5, random_state=123)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_matches_defining_formula_with_noise_removed(self):
        X, y = generate_synthetic_data(n_samples=200, n_features=5, noise_std=0.0, random_state=0)
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        expected = 3 * x1 ** 2 + np.sin(x2) + 0.5 * x3
        np.testing.assert_allclose(y, expected)

    def test_rejects_too_few_features(self):
        with self.assertRaises(ValueError):
            generate_synthetic_data(n_samples=10, n_features=2)

    def test_rejects_non_positive_sample_count(self):
        with self.assertRaises(ValueError):
            generate_synthetic_data(n_samples=0, n_features=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
