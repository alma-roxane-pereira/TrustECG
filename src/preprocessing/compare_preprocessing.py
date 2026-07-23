import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# CHANGE THESE PATHS
# ----------------------------

original_path = r"data/raw/WFDBRecords_9Class/01/010/JS00001.mat"
processed_path = r"data/preprocessed_9Class/01/010/JS00001.npy"

lead = 0  # Lead I

# ----------------------------
# Load ECGs
# ----------------------------

original = sio.loadmat(original_path)["val"]

processed = np.load(processed_path)

# ----------------------------
# Plot
# ----------------------------

start = 1000
end = 2000

plt.figure(figsize=(15, 8))

plt.subplot(2,1,1)
plt.plot(original[lead][start:end])
plt.title("Original ECG (Zoomed)")

plt.subplot(2,1,2)
plt.plot(processed[lead][start:end])
plt.title("Preprocessed ECG (Zoomed)")

plt.tight_layout()
plt.show()



original = sio.loadmat(original_path)["val"][0]
processed = np.load(processed_path)[0]

fs = 500

freq = np.fft.rfftfreq(len(original), d=1/fs)

orig_fft = np.abs(np.fft.rfft(original))
proc_fft = np.abs(np.fft.rfft(processed))

plt.figure(figsize=(12,5))

plt.plot(freq, orig_fft, label="Original")
plt.plot(freq, proc_fft, label="Preprocessed")

plt.xlim(0,100)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()

plt.grid(True)

plt.show()