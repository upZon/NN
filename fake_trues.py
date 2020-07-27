import numpy as np


class biggest_nn:
    def __init__(self, t):
        self.inputs = 1  # UNITS IN INPUT LAYER
        self.outputs = 2  # UNITS IN OUTPUT LAYER

        self.weight = np.random.randn(self.inputs, self.outputs)  # (1x2)
        self.bias = 0
        self.lr = .09  # LEARNING RATE

        # SEPARATE TRAINING DATA INTO FEATURES AND DESIRED 'Y'
        self.features = np.array(t[:, 0], dtype=float).reshape(2, 1)
        desires = t[:, 1]
        self.y = np.array([desires[:][0], desires[:][1]], dtype=float)

    # APPLY SIGMOID ACTIVATION
    def sig(self, z):
        return 1 / (1 + np.e**(-z))

    # FORWARD PROPAGATION
    def forward_props(self, guess=None):
        if guess is not None:  # FOR QUIZ
            features = guess
        else:
            features = self.features

        # features (2x1) @ weight (1x2) ---> (2x2) "a2"
        self.a2 = self.sig(np.dot(features, self.weight) + self.bias)

        return self.a2

    # DEFINE COST FUNCTION (TO MONITOR PROGRESS)
    def cost(self):

        # COST FUNCTION IN ONE PIECE
        # error = (-1 * ((y * np.log(a)) + ((1 - y) * np.log(1 - a)))) / self.m

        error = []  # LIST TO GATHER EACH OUTPUT ERROR

        # LOOP THROUGH COST FOR EACH OUTPUT
        for i in range(len(self.y)):

            # ERROR WHEN Y == 1
            error_class1 = -self.y[i] * np.log(self.a2[i])

            # ERROR WHEN Y == 0
            error_class0 = (1 - self.y[i]) * np.log(1 - self.a2[i])

            # SUM OF BOTH ERRORS
            error.append(sum(error_class1 - error_class0))

        return sum(error) / len(self.y)  # AVERAGE ERROR

    # THE DESCENT
    def back_props(self):

        db = self.a2 - self.y  # "DELTA" (2x2)

        # FEATURES.T (1x2) @ "DELTA" (2x2) ---> "dw" (1x2)
        dw = (np.dot(self.features.T, db)) / len(self.y)
        db = db.sum() / len(self.y)  # PARTIAL BIAS

        # APPLY LEARNING RATE
        dw *= self.lr
        db *= self.lr

        # APPLY GRADIENT DESCENT
        self.weight -= dw
        self.bias -= db

        return

    # APPLY DECISION BOUNDARY (FOR QUIZ)
    def decision_boundary(self, prob):
        return 1 if prob >= 0.5 else 0

    # PREPARE VECTOR FOR DECISION BOUNDARY (FOR QUIZ)
    def flat_it(self, predictions):
        dbound = np.vectorize(self.decision_boundary)  # MAKES FUNCTION A LOOP
        return dbound(predictions).flatten()  # MAKES VECTOR ONE ROW


# GROUND ZERO
if __name__ == '__main__':

    # TRAINING DATA ([(INPUTS, [DESIRED OUTPUTS])])
    trainer = np.array([(1, [0, 0]),
                        (0, [1, 1])])

    bnn = biggest_nn(trainer)  # INITIALIZE THE BIGGEST NEURAL NETWORK

    # QUIZ ARRAYS
    guess1 = np.array([1])
    guess2 = np.array([0])

    # TEST BEFORE LEARNING
    print(bnn.flat_it(bnn.forward_props(guess1)))
    print(bnn.flat_it(bnn.forward_props(guess2)))

    i = 0  # ITERATION COUNTER
    j = 1  # INITIAL COST VARIABLE
    GOAL = .05  # SET COST GOAL

    print()
    print("Learning...")
    print()

    # LEARNING LOOP
    while j > GOAL:
        bnn.forward_props()  # ACTIVATE FORWARD PROPAGATION
        j = bnn.cost()  # UPDATE COST VARIABLE
        if i % 100 == 0:
            print(j)  # PRINT COST AT SPECIFIED ITERATION (TO MONITOR PROGRESS)
        bnn.back_props()  # ACTIVATE BACKWARD PROPAGATION
        i += 1

    print(j)  # PRINT FINAL COST
    print()

    # QUIZ AFTER LEARNING
    print(bnn.flat_it(bnn.forward_props(guess1)))
    print(bnn.flat_it(bnn.forward_props(guess2)))
    print()
