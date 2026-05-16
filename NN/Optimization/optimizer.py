import numpy


class Optimizer:

    def __init__(self, learning_rate: float):

        self.learning_rate = learning_rate

    def gradient_descent(self, weights:numpy.ndarray, gradient: numpy.ndarray) -> numpy.ndarray:

        weights = weights - self.learning_rate * gradient

        return weights



