import matplotlib.pyplot as plt
import numpy

class Visualizer:

    loss_history=[]
    epochs = []


    @classmethod
    def update(cls, loss: float, epoch: int):
        cls.loss_history.append(loss)
        cls.epochs.append(epoch+1)

    @classmethod
    def visualize_loss(cls):

        y = numpy.array(cls.loss_history)
        x = numpy.array(cls.epochs)

        plt.plot(x, y)
        plt.show()
        plt.close()




