import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
from pathlib import Path

def predict(X, w, b):
    m, n = X.shape
    p = np.zeros(m)

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        p[i] = f_wb_i >= 0.5

    return p

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def compute_cost(X, y, w, b, *argv):
    m, n = X.shape

    total_cost = 0.
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        total_cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    total_cost /= m
    return total_cost


def compute_cost_reg(X, y, w, b, lambda_=1):
    m, n = X.shape

    cost_without_reg = compute_cost(X, y, w, b)
    reg_cost = 0.
    for j in range(n):
        w_j = w[j] ** 2
        reg_cost += w_j

    reg_cost = reg_cost * lambda_ / (2 * m)
    total_cost = cost_without_reg + reg_cost

    return total_cost

def compute_gradient(X, y, w, b, *argv):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        error = sigmoid(np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + error * X[i, j]
        dj_db += error
    dj_db /= m
    dj_dw /= m

    return dj_db, dj_dw


def compute_gradient_reg(X, y, w, b, lambda_=1):
    m, n = X.shape

    dj_db, dj_dw = compute_gradient(X, y, w, b)

    ### START CODE HERE ###
    for j in range(n):
        dj_dw[j] += w[j] * (lambda_ / m)
    ### END CODE HERE ###

    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    J_history = []
    w_history = []

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)

        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        if i < 100000:
            cost = cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

    return w_in, b_in, J_history, w_history

X_train, y_train = load_data("data/ex2data2.txt")

# Data Representation
print("X_train:", X_train[:5])
print("Type of X_train:",type(X_train))
print("y_train:", y_train[:5])
print("Type of y_train:",type(y_train))
print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))

# Visualizing the Data
plot_data(X_train, y_train[:], pos_label="Accepted", neg_label="Rejected")
plt.ylabel('Microchip Test 2')
plt.xlabel('Microchip Test 1')
plt.legend(loc="upper right")
plt.show()

# Feature Mapping
print("Original shape of data:", X_train.shape)

mapped_X =  map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", mapped_X.shape)
print("X_train[0]:", X_train[0])
print("mapped X_train[0]:", mapped_X[0])

# Testing Compute Cost Regularization
X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5
lambda_ = 0.5
cost = compute_cost_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print("Regularized cost :", cost)

# Testing Compute Gradient Reg
X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5

lambda_ = 0.5
dj_db, dj_dw = compute_gradient_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print(f"dj_db: {dj_db}", )
print(f"First few elements of regularized dj_dw:\n {dj_dw[:4].tolist()}", )

np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1])-0.5
initial_b = 1.

lambda_ = 0.01

iterations = 10000
alpha = 0.01

w,b, J_history,_ = gradient_descent(X_mapped, y_train, initial_w, initial_b,
                                    compute_cost_reg, compute_gradient_reg,
                                    alpha, iterations, lambda_)

# Plotting the decision boundary
plot_decision_boundary(w, b, X_mapped, y_train)
plt.ylabel('Microchip Test 2')
plt.xlabel('Microchip Test 1')
plt.legend(loc="upper right")
plt.show()

#Compute accuracy on the training set
p = predict(X_mapped, w, b)

print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))