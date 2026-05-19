import numpy


class Optimizer:

    def __init__(self, learning_rate: float):

        self.learning_rate = learning_rate

class GradientDescent(Optimizer):

    def __init__(self, learning_rate: float):
        super().__init__(learning_rate)


    def optimize(self, weights:numpy.ndarray, gradient: numpy.ndarray) -> numpy.ndarray:

        weights = weights - self.learning_rate * gradient

        return weights



