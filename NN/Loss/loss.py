import numpy


class Loss:

    def forward(self, y_out, y_truth):
        return NotImplementedError

    def backward(self, y_out, y_truth):
        return NotImplementedError

class L1(Loss):

    def forward(self, y_out, y_truth):

        loss = - numpy.abs(y_out - y_truth)
        average_loss = numpy.mean(loss)

        return average_loss

    def backward(self, y_out, y_truth):

        dl = None

        dl_1 = numpy.where(y_out > y_truth, 1, -2) # right side of the minimum

        dl_2 = numpy.where(y_out < y_truth, -1,  0) # left side of the minimum

        # merge both
        dl = numpy.where(dl_1 > dl_2, dl_1, dl_2)

        assert dl is not None, "L1 Loss Gradient can not be None!"

        return dl

class BCE(Loss):

    def forward(self, y_out, y_truth):

        loss = - (y_truth * numpy.log(y_out) + (1-y_truth) * (1-numpy.log(y_out))) # binary cross entropy
        average_loss = numpy.mean(loss) # accumulated loss

        return average_loss

    def backward(self, y_out, y_truth):

        # Don`t sum up the gradient, one gradient for each sample.
        # 1/N in front of the loss is necessary to average the loss without changing the learning Rate for every Batch

        dL = -1/len(y_out) * (y_truth/y_out + (1-y_truth)/(1-y_out)) # derivative of dL

        return dL


