#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Generate some data
t = np.linspace(0, 1, 1000)
y1 = np.sin(2*np.pi*10*t)
y2 = np.sin(2*np.pi*20*t)
y3 = np.sin(2*np.pi*30*t)
y4 = np.sin(2*np.pi*40*t)

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2)

# Plot data in each subplot
axs[0, 0].plot(t, y1)
axs[0, 0].set_title('Plot 1')
axs[0, 1].plot(t, y2)
axs[0, 1].set_title('Plot 2')
axs[1, 0].plot(t, y3)
axs[1, 0].set_title('Plot 3')
axs[1, 1].plot(t, y4)
axs[1, 1].set_title('Plot 4')

# Add overall title to the figure
fig.suptitle('Four Subplots')

# Display the figure
plt.show()
