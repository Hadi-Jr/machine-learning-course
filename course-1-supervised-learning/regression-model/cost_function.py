import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('./deeplearning.mplstyle')

def compute_cost(x, y, w, b):
    m = len(x)

    j_wb = 0
    for i in range(m):
        f_wb = x[i] * w + b
        j_wb += (f_wb - y[i])**2
    j_wb /= (1 / (2 * m))
    return j_wb

x_train = np.array([1.0, 2.0])           #(size in 1000 square feet)
y_train = np.array([300.0, 500.0])           #(price in 1000s of dollars)
w = 200
b = 100

cost = compute_cost(x_train, y_train, w, b)

# Example 2 with 3D representations
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])
plt.close('all')
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
soup_bowl()






