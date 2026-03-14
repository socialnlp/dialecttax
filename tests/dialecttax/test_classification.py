"""Tests for dialecttax.classification."""

import numpy as np
import torch

from dialecttax.classification import LogisticRegression, LogisticRegressionPyTorch


##############
# TEST DATA  #
##############

def _separable_data():
    """Two linearly separable clusters in 2D."""
    rng = np.random.RandomState(42)
    X = np.vstack([rng.randn(50, 2) + [2, 2], rng.randn(50, 2) + [-2, -2]]).astype(np.float32)
    y = np.array([0] * 50 + [1] * 50)
    return X, y


def _multiclass_data():
    """Three linearly separable clusters in 4D."""
    rng = np.random.RandomState(0)
    centers = np.array([[3, 3, 0, 0], [-3, -3, 0, 0], [0, 0, 3, 3]], dtype=np.float32)
    X = np.vstack([rng.randn(40, 4).astype(np.float32) + c for c in centers])
    y = np.array([0] * 40 + [1] * 40 + [2] * 40)
    return X, y


######################
# LOGISTIC REGRESSION #
######################

class TestLogisticRegression:
    def test_fit_and_predict_shape(self):
        X, y = _separable_data()
        clf = LogisticRegression(classes=np.unique(y))
        clf.train(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_separable_accuracy(self):
        X, y = _separable_data()
        clf = LogisticRegression(classes=np.unique(y))
        clf.train(X, y)
        acc = (clf.predict(X) == y).mean()
        assert acc > 0.95

    def test_predict_probs_shape(self):
        X, y = _separable_data()
        clf = LogisticRegression(classes=np.unique(y))
        clf.train(X, y)
        proba = clf.predict_probs(X)
        assert proba.shape == (100, 2)

    def test_predict_probs_sums_to_one(self):
        X, y = _separable_data()
        clf = LogisticRegression(classes=np.unique(y))
        clf.train(X, y)
        proba = clf.predict_probs(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_multiclass(self):
        X, y = _multiclass_data()
        clf = LogisticRegression(classes=np.unique(y))
        clf.train(X, y)
        acc = (clf.predict(X) == y).mean()
        assert acc > 0.90
        assert clf.predict_probs(X).shape == (120, 3)


##############################
# LOGISTIC REGRESSION PYTORCH #
##############################

class TestLogisticRegressionPyTorch:
    def test_fit_and_predict_shape(self):
        X, y = _separable_data()
        clf = LogisticRegressionPyTorch(dim=2, n_classes=2, device="cpu")
        clf.train(X, y, n_epochs=200)
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_separable_accuracy(self):
        X, y = _separable_data()
        clf = LogisticRegressionPyTorch(dim=2, n_classes=2, device="cpu")
        clf.train(X, y, n_epochs=500)
        acc = (clf.predict(X) == y).mean()
        assert acc > 0.95

    def test_predict_probs_shape(self):
        X, y = _separable_data()
        clf = LogisticRegressionPyTorch(dim=2, n_classes=2, device="cpu")
        clf.train(X, y, n_epochs=200)
        proba = clf.predict_probs(X)
        assert proba.shape == (100, 2)

    def test_predict_probs_sums_to_one(self):
        X, y = _separable_data()
        clf = LogisticRegressionPyTorch(dim=2, n_classes=2, device="cpu")
        clf.train(X, y, n_epochs=200)
        proba = clf.predict_probs(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_train_returns_losses(self):
        X, y = _separable_data()
        clf = LogisticRegressionPyTorch(dim=2, n_classes=2, device="cpu")
        losses = clf.train(X, y, n_epochs=50)
        assert len(losses) == 50
        assert losses[-1] < losses[0]

    def test_early_stopping(self):
        X, y = _separable_data()
        clf = LogisticRegressionPyTorch(dim=2, n_classes=2, device="cpu")
        losses = clf.train(X, y, n_epochs=5000, patience=10)
        assert len(losses) < 5000

    def test_early_stopping_disabled(self):
        X, y = _separable_data()
        clf = LogisticRegressionPyTorch(dim=2, n_classes=2, device="cpu")
        losses = clf.train(X, y, n_epochs=50, patience=0)
        assert len(losses) == 50

    def test_multiclass(self):
        X, y = _multiclass_data()
        clf = LogisticRegressionPyTorch(dim=4, n_classes=3, device="cpu")
        clf.train(X, y, n_epochs=500)
        acc = (clf.predict(X) == y).mean()
        assert acc > 0.90
        assert clf.predict_probs(X).shape == (120, 3)
