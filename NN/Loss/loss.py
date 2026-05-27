from typing import Any
import numpy as np
from abc import ABC, abstractmethod



class Loss(ABC):
    """
    Base class for loss functions.
    """

    @staticmethod
    @abstractmethod
    def forward(y_out: np.ndarray, y_truth: np.ndarray) -> float:
        """
        Compute the loss value.

        Args:
            y_out: Predicted output from the network.
            y_truth: Ground truth labels or target values.

        Returns:
            Computed loss value.
        """
        pass

    @staticmethod
    @abstractmethod
    def backward(y_out: np.ndarray, y_truth: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss function.

        Args:
            y_out: Predicted output from the network.
            y_truth: Ground truth labels or target values.

        Returns:
            Gradient of the loss with respect to y_out.
        """
        pass

class L1(Loss):
    """
    Mean Absolute Error (L1) loss.
    Minimizes around the median.
    """

    @staticmethod
    def forward(y_out: np.ndarray, y_truth: np.ndarray) -> float:
        loss = np.abs(y_out - y_truth)
        return float(np.mean(loss))

    @staticmethod
    def backward(y_out: np.ndarray, y_truth: np.ndarray) -> np.ndarray:
        return np.sign(y_out - y_truth)

class L2(Loss):
    """
    Mean Squared Error (L2) loss.
    Minimizes around the mean.
    """

    @staticmethod
    def forward(y_out: np.ndarray, y_truth: np.ndarray) -> float:
        pass

    @staticmethod
    def backward(y_out: np.ndarray, y_truth: np.ndarray) -> np.ndarray:
        pass

class BCE(Loss):
    """
    Binary Cross Entropy loss.
    """

    @staticmethod
    def forward(y_out: np.ndarray, y_truth: np.ndarray) -> float:

        y_out = np.clip(y_out, 1e-7, 1 - 1e-7)

        loss = - (y_truth * np.log(y_out) + (1-y_truth) * (np.log(1-y_out))) # binary cross entropy
        average_loss = np.mean(loss) # accumulated loss

        return float(average_loss)

    @staticmethod
    def backward(y_out: np.ndarray, y_truth: np.ndarray) -> np.ndarray:

        # Don`t sum up the gradient, one gradient for each sample.
        # 1/N in front of the loss is necessary to average the loss without changing the learning Rate for every Batch

        dl = -1/len(y_out) * (y_truth/y_out - (1-y_truth)/(1-y_out)) # derivative of dL

        return dl



