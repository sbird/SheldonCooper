import numpy as np
import matplotlib.pyplot as plt
x = np.array([1, 2, 3, 4, 5])
a =2.5
y = a*x + np.random.normal(0, 1, size=x.shape)

plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of y = 2.5x + noise')
plt.grid()
plt.show()