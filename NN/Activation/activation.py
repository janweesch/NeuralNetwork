import numpy
from abc import ABC, abstractmethod

class ActivationFunction(ABC):

    @staticmethod
    @abstractmethod
    def forward(z_input: numpy.ndarray):
        pass

    @staticmethod
    @abstractmethod
    def backward(z_input: numpy.ndarray):
        pass

class Affine(ActivationFunction):

    @staticmethod
    def forward(z_input: numpy.ndarray):
        return z_input

    @staticmethod
    def backward(z_input: numpy.ndarray):
        return numpy.ones((z_input.shape[0], 1))

class Sigmoid(ActivationFunction):

    @staticmethod
    def forward(z_input: numpy.ndarray):

        a = 1 / (1 + numpy.exp(-z_input))
        return a

    @staticmethod
    def backward(z_input: numpy.ndarray):

        dz = 1 / (1 + numpy.exp(-z_input)) * (1 - (1 / (1 + numpy.exp(-z_input))))

        return dz

class ReLU(ActivationFunction):

    @staticmethod
    def forward(z_input: numpy.ndarray):

        a = numpy.maximum (0, z_input)

        return a

    @staticmethod
    def backward(z_input: numpy.ndarray):

        dz = numpy. where(z_input > 0, 1, 0)

        return dz

class Quadratic:

    @staticmethod
    def forward(z_input: numpy.ndarray):
        a = z_input**2

        return a

    @staticmethod
    def backward(z_input:numpy.ndarray):
        dz = 2*z_input

        return dz
