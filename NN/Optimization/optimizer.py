from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):
    """
    Base class for optimizers that update neural network parameters
    using gradients computed during backpropagation.

    Attributes:
          learning_rate: Step size in the direction of the gradient (how much to trust the gradient)

    Args:
          learning_rate: Step size in the direction of the gradient (how much to trust the gradient)
    """

    def __init__(self, learning_rate: float = 0.05):

        self.learning_rate = learning_rate

    @abstractmethod
    def optimize(self, weights:np.ndarray, gradient:np.ndarray) -> np.ndarray:
        """
        Updates the weights in each layer according to the gradient.

        Args:
            weights: Weights of the current layer
            gradient: Gradient of the loss with respect to weights
        """
        pass


class GradientDescent(Optimizer):
    """
    Basic Gradient Descent optimizer.
    """

    def __init__(self, learning_rate: float=0.1):
        super().__init__(learning_rate)


    def optimize(self, weights:np.ndarray, gradient: np.ndarray) -> np.ndarray:

        return weights - self.learning_rate * gradient
