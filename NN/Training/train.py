from NN.Loss.loss import Loss, BCE
from NN.Network.network import Network
from NN.Optimization.optimizer import Optimizer




class Train:

    def __init__(self, network: Network, loss: Loss, optimizer: Optimizer):

        self.network = network
        self.optimizer = optimizer
        self.loss = loss

    def train(self, training_data, y_truth, epochs: int, batch_size: int):

        for epoch in range(epochs):

            f_output = self.network.forward(training_data) # network forward pass
            f_loss = self.loss.forward(self.loss, f_output, y_truth) # calculate loss

            b_loss = self.loss.backward(f_loss, y_truth)
            self.network.backward(b_loss, optimizer=self.optimizer)

        return self.network












