from typing import Optional

from NN.Activation.activation import Sigmoid, ActivationFunction
from NN.Loss.loss import Loss, BCE
from NN.Network.network import Network
from NN.Optimization.optimizer import Optimizer
from NN.Loss.visualizer import Visualizer




class Train:

    def __init__(self, network: Network, loss: Loss, optimizer: Optimizer):

        self.network = network
        self.optimizer = optimizer
        self.loss = loss

    def train(self, training_data, y_truth, epochs: int, batch_size: Optional[int]=None):

        for epoch in range(epochs):

            y_out = self.network.forward(training_data) # network forward pass (inclusive sigmoid function)

            f_loss = self.loss.forward(y_out=y_out, y_truth=y_truth) # calculate loss)
            Visualizer.update(f_loss, epoch)

            b_loss = self.loss.backward(y_out, y_truth)
            self.network.backward(b_loss, optimizer=self.optimizer)

        Visualizer.visualize_loss()


        return self.network












