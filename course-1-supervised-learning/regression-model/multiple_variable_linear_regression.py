import copy, math
import numpy as np
import matplotlib.pyplot as plt

############### LINEAR REGRESSION WITH MULTIPLE VARIABLES MODEL ###############

def predict_single_loop(x, w, b):
    n = x.shape[0]
    p = 0

    for i in range(n):
        p_i = x[i] * w[i]
        p = p + p_i

    return p + b

def predict(x, w, b):
    p = np.dot(x, w) + b
    return p

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0

    for i in range(m):
        fw_b_i = np.dot(X[i], w) + b
        cost += (fw_b_i - y[i]) ** 2
    cost /= (2 * m)
    return cost

def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        # j because X is a vector and we need to multiply a scalar with a vector - result is a vector
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err
    dj_dw /= m
    dj_db /= m

    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(cost_function(X, y, w, b))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history

X_train = np.array(
    [
        [2104, 5, 1, 45],
        [1416, 3, 2, 40],
        [852, 2, 1, 35]
    ]
)

y_train = np.array(
    [460, 232, 178]
)

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
initial_w = np.zeros_like(w_init)
initial_b = 0.
iterations = 1000
alpha = 5.0e-7
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient,
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m = X_train.shape[0]
for i in range(m):
    # f_wb(X) = W . X + b
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")