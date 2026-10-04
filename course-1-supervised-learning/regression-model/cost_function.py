import numpy as np
import matplotlib.pyplot as plt

def compute_cost(x, y, w, b):
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        fw_b = w * x[i] + b
        cost = (fw_b - y[i]) ** 2
        cost_sum+= cost
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost

def compute_output_model(sizes, w, b):
    fw_b = np.zeros(len(sizes))
    for i in range(len(sizes)):
        fw_b[i] = w * sizes[i] + b
    return fw_b

#(size in 1000 square feet)
sizes = np.array([1.0, 2.0])
#(price in 1000s of dollars)
prices = np.array([300.0, 500.0])

#parameters
b = 100
w = 200

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))

# housing prices plot
predictions = compute_output_model(sizes, w, b)
ax1.plot(sizes, predictions, c='b', linewidth=3)
ax1.scatter(sizes, prices, marker='x', c='r', linewidth=3, s=80)
for x, y_actual, y_pred in zip(sizes, prices, predictions):
    ax1.plot([x, x], [y_actual, y_pred], linestyle='--', c='gray', linewidth=3)

#cost vs w plot
w_values = np.linspace(0, 400, 200)
cost_values = []
for w_value in w_values:
    cost_values.append(compute_cost(sizes, prices, w_value, b))
ax2.plot(w_values, cost_values)
cost_function = compute_cost(sizes, prices, w, b)
ax2.plot(w, cost_function, "ro")
ax2.set_xlabel("w")
ax2.set_ylabel("Cost")

plt.show()