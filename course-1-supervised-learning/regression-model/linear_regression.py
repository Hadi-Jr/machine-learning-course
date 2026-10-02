import numpy as np
import matplotlib.pyplot as plt

##################### LINEAR REGRESSION WITH ONE VARIABLE MODEL #####################

house_sizes = np.array([70.0, 80.0, 90.0, 100.0, 150.0, 200.0]) #in square meters
prices = np.array([140000, 160000, 180000, 200000, 300000, 400000]) #in dollars

def compute_output_model(input_data, w, b):
    input_length = len(input_data)
    fw_b = np.zeros(input_length)

    for i in range(input_length):
        fw_b[i] = w * input_data[i] + b

    return fw_b

def predict_price(size, w, b):
    fw_b = w * size + b
    return fw_b

w = 2000
b = 0

tmp_f_wb = compute_output_model(house_sizes, w, b)

prediction = predict_price(66, w, b)
print(prediction)

plt.plot(house_sizes, tmp_f_wb, c='b', label='Our Prediction')
plt.scatter(house_sizes, prices, marker='x', c='r', label='Actual Values')

plt.title("Housing Prices")
plt.ylabel('Prices')
plt.xlabel('Sizes')
plt.legend()
plt.show()

