import numpy as np
np.seterr(divide='ignore')
np.seterr(invalid='ignore')


# APPLY SIG
def sig(z):
    return 1 / (1 + (np.e**(-z)))


# DEFINE Z
def z(x, w, b):
    # x (4, 2) @ w (2, 1) --> (4, 1)
    return sig(np.dot(x, w) + b)


# APPLY DECISION BOUNDARY
def decision_boundary(prob):
    return 1 if prob >= 0.5 else 0


# PREPARE VECTOR FOR DECISION BOUNDARY
def flat_it(predictions):
    dbound = np.vectorize(decision_boundary)  # makes function a loop
    return dbound(predictions).flatten()  # makes vector 1 row


# DEFINE COST FUNCTION
def cost(a, y, m):

    # COST FUNCTION IN ONE PIECE
    # cost = -1 * ((y * np.log(a)) + ((1 - y) * np.log(1 - a)))

    # ERROR WHEN Y == 1
    cost_class1 = -y * np.log(a)

    # ERROR WHEN Y == 0
    cost_class0 = (1 - y) * np.log(1 - a)

    # SUM OF BOTH ERRORS
    cost = cost_class1 - cost_class0

    # TAKE THE AVERAGE ERROR
    cost = cost.sum() / m

    return cost


# APPLY GRADIENT DESCENT
def grad(a, x, y, w, b, m, lr):

    # a = (4, 1)
    # x = (4, 2)
    # y = (4, 1)
    # w = (2, 1)

    # dw = x.T (2, 4) @ [a - y] (4, 1) ---> (2, 1)

    dw = np.dot(x.T, (a - y))
    dw /= m

    db = np.sum(a - y)
    db /= m

    dw *= lr
    db *= lr

    w = w - dw  # w (2, 1) - dw (2, 1)
    b = b - db

    return w, b


# DEFINE INPUTS, OUTPUTS, WEIGHTS, BIAS
# x = (4, 2) (m, f)
# y = (4, 1) (m, o)
# w = (2, 1) (o, f)

x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([0, 0, 0, 1]).reshape(4, 1)
w = np.zeros((2, 1))  # INITIAL WEIGHTS
b = 0  # INITIAL BIAS
m = np.size(y)  # TOTAL NUMBER OF ROWS
GOAL = 0.009
lr = .7

# a = flat_it(a).reshape(x_shape)

guess = np.array([0, 1])

print("Initial guess:", 1 if z(guess, w, b) >= .5 else 0)
print()
print("Training...")
print()

# LEARNING LOOP
a = z(x, w, b)
j = cost(a, y, m)
i = 0
while j > GOAL:
    a = z(x, w, b)
    if i % 1000 == 0:  # PLOT GRAPH AT SPECIFIED ITERATION
        print(j)
        # pass
    elif j < GOAL:
        print(j)
        break
    w, b = grad(a, x, y, w, b, m, lr)
    j = cost(a, y, m)
    i += 1
    # print(j)

print(" ")
print(f'The minimum occurs at {i}')
print(f'The weight is now {w}')
print(f'The bias is now {b}\n')


print("New guess:", 1 if z(guess, w, b) >= .5 else 0)
print()
