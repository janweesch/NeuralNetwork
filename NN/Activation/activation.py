import numpy
from abc import ABC, abstractmethod

class ActivationFunction(ABC):

    @staticmethod
    @abstractmethod
    def forward(z_input: numpy.ndarray):
        pass

    @staticmethod
    @abstractmethod
    def backward(a_output: numpy.ndarray):
        pass

class Sigmoid(ActivationFunction):

    @staticmethod
    def forward(z_input: numpy.ndarray):

        a = 1 / (1 + numpy.exp(-z_input))
        return a

    @staticmethod
    def backward(a_output: numpy.ndarray):

        dz = 1 / (1 + numpy.exp(-a_output)) * (1 - 1 / (1 + numpy.exp(-a_output)))

        return dz


