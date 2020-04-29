import matplotlib.pyplot as plt
import numpy as np
import mpld3

# Scatter points
fig, ax = plt.subplots()
np.random.seed(0)
x, y = np.random.normal(size=(2, 200))
color, size = np.random.random((2, 200))

ax.scatter(x, y, c=color, s=500 * size, alpha=0.3)
ax.grid(color='lightgray', alpha=0.7)

plt.show(fig)
# s.plot.bar()
fig.savefig('my_plot.png')

# mpld3.display(fig)