import matplotlib.pyplot as plt
import numpy as np

# 1. Setup Parameters
sampling_rate = 1000  # Hz
duration = 1.0  # seconds
t = np.linspace(0, duration, sampling_rate)

# 2. Create Time Domain Signal
# A 10Hz wave and a 50Hz wave combined
signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

# 3. Perform FFT (Move to Frequency Domain)
fft_values = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), 1 / sampling_rate)

# We only care about the positive frequencies for visualization
positive_mask = frequencies > 0
freqs = frequencies[positive_mask]
magnitude = np.abs(fft_values[positive_mask])

# 4. Visualization
plt.figure(figsize=(12, 5))

# Time Domain Plot
plt.subplot(1, 2, 1)
plt.plot(t[:100], signal[:100])  # Plot first 100ms for clarity
plt.title("Time Domain (Amplitude vs Time)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

# Frequency Domain Plot
plt.subplot(1, 2, 2)
plt.stem(freqs[:60], magnitude[:60])
plt.title("Frequency Domain (Magnitude vs Frequency)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Strength")

plt.tight_layout()
plt.show()
