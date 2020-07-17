import numpy as np


class NN:
    def __init__(self, x, y, m):
        # DEFINE UNITS IN LAYERS
        self.input_units = 2
        self.hidden_units = 2
        self.output_units = 1

        # VARIABLIZE (yep) INPUTS, OUTPUTS, # OF ROWS
        self.features = x  # (4x2)
        self.y = y  # (4x1)
        self.m = m

        # RANDOMIZE WEIGHTS (w1 --> (2x2), w2 --> (2x1))
        self.w1 = np.random.randn(self.input_units, self.hidden_units)
        self.w2 = np.random.randn(self.hidden_units, self.output_units)

        # INITIALIZE LEARING RATE (ALPHA)
        self.lr = 3

    # APPLY SIGMOID
    def sig(self, feets, weights):
        return 1 / (1 + np.e**(-(np.dot(feets, weights))))

    # APPLY SIGMOID PRIME
    def sig_dash(self, a):
        return a * (1 - a)

    # DEFINE COST FUNCTION (TO MONITOR PROGRESS)
    def cost(self):

        # COST FUNCTION IN ONE PIECE
        # error = -1 * ((y * np.log(a)) + ((1 - y) * np.log(1 - a)))

        # ERROR WHEN Y == 1
        error_class1 = -y * np.log(self.a3)

        # ERROR WHEN Y == 0
        error_class0 = (1 - y) * np.log(1 - self.a3)

        # SUM OF BOTH ERRORS
        error = error_class1 - error_class0

        # TAKE THE AVERAGE ERROR
        self.error = error.sum() / self.m

        return self.error

    # FORWARD PROPAGATION
    def forward_props(self):

        # *** NOTE: THE INPUTS ARE LAYER 1 *** #

        # ACTIVATE ON LAYER 2
        # features (4x2) @ w1 (2x2) ---> (4x2) "a2"
        self.a2 = self.sig(self.features, self.w1)

        # ACTIVATE ON LAYER 3
        # a2 (4x2) @ w2 (2x1) ---> (4x1) "a3"
        self.a3 = self.sig(self.a2, self.w2)

        return self.cost()

    # BACK PROPAGATION (Deep breath...)
    def back_props(self):

        # DEFINE DELTA3
        # a3 (4x1) - y (4x1) ---> (4x1) "loss"
        # a3 (4x1) * [1 - a3] (4x1) ---> (4x1) "sigdiv3"
        # loss (4x1) * sigdiv3 (4x1) ---> (4x1) "delta3"
        self.delta3 = np.multiply((self.a3 - self.y), self.sig_dash(self.a3))

        # GET DERIAVATIVE OF W2
        # a2.T (2x4) @ delta3 (4x1) ---> (2x1) "dw2"
        self.dw2 = np.dot(self.a2.T, self.delta3)

        # DEFINE DELTA2
        # delta3 (4x1) @ w2.T (1x2) ---> (4x2) "hubba hubba"
        # a2 (4x2) * [1 - a2] (4x2) ---> (4x2) "sigdiv2"
        # hubba hubba (4x2) * sigdiv2 (4x2) ---> (4x2) "delta2"
        self.delta2 = (np.dot(self.delta3, self.w2.T) * self.sig_dash(self.a2))

        # GET DERIAVATIVE OF W1
        # x.T (2x4) @ delta2 (4x2) ---> (2x2) "dw1"
        self.dw1 = np.dot(self.features.T, self.delta2)

        # APPLY LEARNING RATE
        self.dw2 *= self.lr
        self.dw1 *= self.lr

        # UPDATE WEIGHTS
        self.w2 -= self.dw2
        self.w1 -= self.dw1

        return


# GROUND ZERO
if __name__ == '__main__':

    # DEFINE INPUTS, OUTPUTS, # OF ROWS
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 2)
    y = np.array([1, 0, 0, 1]).reshape(4, 1)
    m = np.size(y)

    nn = NN(x, y, m)  # INITIATE NEURAL NETWORK

    # LEARNING LOOP
    for i in range(10000):
        j = nn.forward_props()  # FORWARD PROPAGATION
        if i % 1000 == 0:  # PRINT COST AT SPECIFIED ITERATION
            print(j)
        nn.back_props()  # BACK PROPAGATION
