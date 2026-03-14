import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


##################
# CLASSIFICATION #
##################

def probs_to_class(probs: "np.ndarray") -> "np.ndarray":
    """Convert probability matrix to predicted class labels via argmax.

    Args:
        probs: Probability matrix of shape (N, n_classes).

    Returns:
        Predicted class labels of shape (N,).
    """
    return np.argmax(probs, axis=1)

class LogisticRegression:
    """Logistic regression via SGD with feature scaling.

    Wraps sklearn's SGDClassifier with log_loss (equivalent to logistic regression)
    and StandardScaler preprocessing.

    Args:
        max_iter: Maximum number of passes over the training data.
    """
    def __init__(self, classes: "np.ndarray", max_iter: int = 1000):
        self.classes = np.asarray(classes)
        self.scaler = StandardScaler()
        self.classifier = SGDClassifier(loss="log_loss", max_iter=max_iter, n_jobs=-1)

    def train(self, X: "np.ndarray", y: "np.ndarray") -> None:
        """Fit the scaler and classifier using partial_fit for mini-batch support.

        Args:
            X: Feature matrix of shape (N, D).
            y: Integer class labels of shape (N,).
        """
        X_scaled = self.scaler.partial_fit(X).transform(X)
        self.classifier.partial_fit(X_scaled, y, classes=self.classes)

    def predict(self, X: "np.ndarray") -> "np.ndarray":
        """Predict class labels.

        Args:
            X: Feature matrix of shape (N, D).

        Returns:
            Predicted class labels of shape (N,).
        """
        return probs_to_class(self.predict_probs(X))

    def predict_probs(self, X: "np.ndarray") -> "np.ndarray":
        """Predict class probabilities.

        Args:
            X: Feature matrix of shape (N, D).

        Returns:
            Probability matrix of shape (N, n_classes).
        """
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)


class LogisticRegressionPyTorch:
    """Logistic regression implemented in PyTorch.

    Single linear layer with cross-entropy loss, trained with Adam.

    Args:
        dim: Input feature dimension.
        n_classes: Number of output classes.
        lr: Learning rate for Adam optimizer.
        device: Device to use. Defaults to CUDA if available.
    """
    def __init__(self, dim: int, n_classes: int, lr: float = 1e-3, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = nn.Linear(dim, n_classes).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(
        self, X: "np.ndarray", y: "np.ndarray",
        n_epochs: int = 1000, patience: int = 10, tol: float = 1e-4,
    ) -> list[float]:
        """Train the model.

        Args:
            X: Feature matrix of shape (N, D).
            y: Integer class labels of shape (N,).
            n_epochs: Maximum number of training epochs.
            patience: Stop after this many epochs without improvement. 0 disables early stopping.
            tol: Minimum loss decrease to count as an improvement.

        Returns:
            List of per-epoch loss values.
        """
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)

        self.model.train()
        losses = []
        best_loss = float("inf")
        wait = 0
        for _ in range(n_epochs):
            logits = self.model(X_t)
            loss = self.loss_fn(logits, y_t)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            cur_loss = loss.item()
            losses.append(cur_loss)

            if patience > 0:
                if cur_loss < best_loss - tol:
                    best_loss = cur_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
        return losses

    @torch.no_grad()
    def predict(self, X: "np.ndarray") -> "np.ndarray":
        """Predict class labels.

        np.argmax(probs, axis=1)

        Args:
            X: Feature matrix of shape (N, D).

        Returns:
            Predicted class labels of shape (N,).
        """
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        logits = self.model(X_t)
        return logits.argmax(dim=1).cpu().numpy()

    @torch.no_grad()
    def predict_probs(self, X: "np.ndarray") -> "np.ndarray":
        """Predict class probabilities.

        Args:
            X: Feature matrix of shape (N, D).

        Returns:
            Probability matrix of shape (N, n_classes).
        """
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        logits = self.model(X_t)
        return torch.softmax(logits, dim=1).cpu().numpy()
