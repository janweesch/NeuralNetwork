import numpy as np
from abc import ABC, abstractmethod

class ActivationFunction(ABC):

    @staticmethod
    @abstractmethod
    def forward(z_input: np.ndarray):
        pass

    @staticmethod
    @abstractmethod
    def backward(z_input: np.ndarray):
        pass

class Affine(ActivationFunction):

    @staticmethod
    def forward(z_input: np.ndarray):
        return z_input

    @staticmethod
    def backward(z_input: np.ndarray):
        return np.ones((z_input.shape[0], 1))

class Sigmoid(ActivationFunction):

    @staticmethod
    def forward(z_input: np.ndarray):

        a = 1 / (1 + np.exp(-z_input))
        return a

    @staticmethod
    def backward(z_input: np.ndarray):

        dz = 1 / (1 + np.exp(-z_input)) * (1 - (1 / (1 + np.exp(-z_input))))

        return dz

class ReLU(ActivationFunction):

    @staticmethod
    def forward(z_input: np.ndarray):

        a = np.maximum (0, z_input)

        return a

    @staticmethod
    def backward(z_input: np.ndarray):

        dz = np.where(z_input > 0, 1, 0)

        return dz
    
class Tanh(ActivationFunction):
    
    @staticmethod
    def forward(z_input: np.ndarray):
        
        a = (np.exp(z_input)-np.exp(-z_input)) / (np.exp(z_input)+np.exp(-z_input))

        return a

    @staticmethod
    def backward(z_input):

        dz = 1 - np.pow((np.exp(z_input)-np.exp(-z_input)) / (np.exp(z_input)+np.exp(-z_input)), 2)

        return dz

class Quadratic:

    @staticmethod
    def forward(z_input: np.ndarray):

        a = z_input**2

        return a

    @staticmethod
    def backward(z_input:np.ndarray):

        dz = 2*z_input

        return dz
