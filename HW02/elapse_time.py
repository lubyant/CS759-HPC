# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

# %%
elapsed_times = []
with open("elapse_time.txt", "r") as f:
    for line in f:
        elapsed_times.append(np.log2(float(line)))
    
# %%
plt.figure()
plt.plot(range(10,31), elapsed_times)
plt.xticks([10,15,20,25,30],[10,15,20,25,30])
plt.xlabel("Scan array len N in log2 scale")
plt.ylabel("Elapse time in log2 scale (millisecond)")
plt.savefig("task1.pdf")
# %%
img = np.array([[1,3,4,5,6],[2,3,5,6,2],
[12,3,4,6,1], [1,3,4,1,3], [6,3,5,6,9]])
kernel = np.array([[1,2,3],[4,5,6], [7,8,9]])
convolve2d(img, kernel, mode="same")
# %%
