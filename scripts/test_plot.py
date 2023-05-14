#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Generate random data
np.random.seed(0)
t = np.linspace(0, 1, 1000)
x = np.random.randn(1000)

# Apply a Butterworth low-pass filter with cutoff frequency 10 Hz
fs = 1000  # Sample rate
fc = 10    # Cutoff frequency
order = 4  # Filter order
Wn = 2*fc/fs
b, a = signal.butter(order, Wn, 'lowpass')
y = signal.filtfilt(b, a, x)

# Plot the original and filtered signals
fig, ax = plt.subplots()
ax.plot(t, x, label='Original signal')
ax.plot(t, y, label='Filtered signal')
ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Original and filtered signals')
plt.show()

