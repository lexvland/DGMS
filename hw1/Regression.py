

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('hw1/dataset.csv')

X = dataset['X'].values
Y = dataset['Y'].values

mean_X = np.mean(X)
mean_Y = np.mean(Y)

print(f"Mean of X: {mean_X}")
print(f"Mean of Y: {mean_Y}")

SS_xy = sum((X - mean_X) * (Y - mean_Y))
SS_xx = sum((X - mean_X)**2)
b1 = SS_xy / SS_xx

b0 = mean_Y - b1 * mean_X

print(f"Slope (b1): {b1}")
print(f"Intercept (b0): {b0}")

plt.scatter(X, Y, color='blue', label='Actual data')

Y_pred = b0 + b1 * X

plt.plot(X, Y_pred, color='red', label='Regression line')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
