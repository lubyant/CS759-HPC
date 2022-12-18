# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
elapsed_times_512 = []
with open("512.out", "r") as f:
    for line in f:
        elapsed_times_512.append(np.log2(float(line)))

elapsed_times_16 = []
with open("16.out", "r") as f:
    for line in f:
        elapsed_times_16.append(np.log2(float(line)))
    
# %%
plt.figure()
plt.plot(range(10,30), elapsed_times_512, label="thread:512", marker='*')
plt.plot(range(10,30), elapsed_times_16, label="thread:16", marker='*')
plt.xticks([10,15,20,25,29],[10,15,20,25,29])
plt.xlabel("Vscale array len N in log2 scale")
plt.ylabel("Elapse time in log2 scale (millisecond)")
plt.legend()
plt.savefig("task3.pdf")

# %%
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
B = np.array([[9,8,7],[6,5,4], [3,2,1]])
print(np.matmul(A, B))
# %%
