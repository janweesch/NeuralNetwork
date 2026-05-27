import matplotlib.pyplot as plt
import numpy
from typing import Optional

class Visualizer:

    @staticmethod
    def visualize_2d_points(data: numpy.ndarray, title: Optional[str]="Scatter Plot"):

        plt.scatter(data[:2, 0], data[:2, 1], color='blue', label='Class 0')
        plt.scatter(data[2:, 0], data[2:, 1], color='red', label='Class 1')
        plt.title(title)
        plt.xlabel("Dim x_1")
        plt.ylabel("Dim x_2")
        plt.legend(loc=9)
        plt.show()


    @staticmethod
    def plot_prediction(data: numpy.ndarray, pred: numpy.ndarray, title: Optional[str]="Plot"):

        plt.scatter(data[:][0], data[:][1], color='blue', label='Ground Truth')
        plt.plot(data[:][0], data[:][1], color='blue', label='Ground Truth')



