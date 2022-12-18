# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
elapsed_times_1024 = []
with open("task1_1024.txt", "r") as f:
    for line in f:
        elapsed_times_1024.append(np.log2(float(line)))

elapsed_times_256 = []
with open("task1_256.txt", "r") as f:
    for line in f:
        elapsed_times_256.append(np.log2(float(line)))

elapsed_times_512 = []
with open("task1_512.txt", "r") as f:
    for line in f:
        elapsed_times_512.append(np.log2(float(line)))

elapsed_times_128 = []
with open("task1_128.txt", "r") as f:
    for line in f:
        elapsed_times_128.append(np.log2(float(line)))
    
# %%
plt.figure()
plt.plot(range(5,15), elapsed_times_512, label="thread:512", marker='*')
plt.plot(range(5,15), elapsed_times_1024, label="thread:1024", marker='*')
plt.plot(range(5,15), elapsed_times_128, label="thread:128", marker='*')
plt.plot(range(5,15), elapsed_times_256, label="thread:256", marker='*')
plt.xticks([5,7,10,12,14],[5,7,10,12,14])
plt.xlabel("Matrix size N in log2 scale")
plt.ylabel("Elapse time in log2 scale (millisecond)")
plt.legend()
plt.savefig("task1.png")

# %%
elapsed_times_1024 = []
with open("task2_1024.txt", "r") as f:
    for line in f:
        elapsed_times_1024.append(np.log2(float(line)))

elapsed_times_256 = []
with open("task2_256.txt", "r") as f:
    for line in f:
        elapsed_times_256.append(np.log2(float(line)))

elapsed_times_512 = []
with open("task2_512.txt", "r") as f:
    for line in f:
        elapsed_times_512.append(np.log2(float(line)))

elapsed_times_128 = []
with open("task2_128.txt", "r") as f:
    for line in f:
        elapsed_times_128.append(np.log2(float(line)))
    
# %%
plt.figure()
plt.plot(range(10,30), elapsed_times_512, label="thread:512", marker='*')
plt.plot(range(10,30), elapsed_times_1024, label="thread:1024", marker='*')
# plt.plot(range(10,30), elapsed_times_128, label="thread:128", marker='*')
plt.plot(range(10,30), elapsed_times_256, label="thread:256", marker='*')
plt.xticks([10,15,20,25,29],[10,15,20,25,29])
plt.xlabel("Vector size N in log2 scale")
plt.ylabel("Elapse time in log2 scale (millisecond)")
plt.legend()
plt.savefig("task2.pdf")
# %%
# bonus 1
elapsed_times_sq = []
with open("matmul.txt", "r") as f:
    for line in f:
        elapsed_times_sq.append(np.log2(float(line)))

elapsed_times_1024 = []
with open("task1_1024.txt", "r") as f:
    for line in f:
        elapsed_times_1024.append(np.log2(float(line)))

plt.figure()
plt.plot(range(5, len(elapsed_times_sq)+5), elapsed_times_sq, label="CPU", marker='*')
plt.plot(range(5, len(elapsed_times_1024)+5), elapsed_times_1024, label="GPU", marker='*')
plt.xticks([5,7,10,12,14],[5,7,10,12,14])
plt.xlabel("Matrix size N in log2 scale")
plt.ylabel("Elapse time in log2 scale (millisecond)")
plt.legend()
plt.savefig("bonus1.pdf")
# %%
# bonus 2
elapsed_times_sq = []
with open("stencil.txt", "r") as f:
    for line in f:
        elapsed_times_sq.append(np.log2(float(line)))

elapsed_times_1024 = []
with open("task2_1024.txt", "r") as f:
    for line in f:
        elapsed_times_1024.append(np.log2(float(line)))

plt.figure()
plt.plot(range(10, len(elapsed_times_sq)+10), elapsed_times_sq, label="CPU", marker='*')
plt.plot(range(10, len(elapsed_times_1024)+10), elapsed_times_1024, label="GPU", marker='*')
plt.xticks([10,15,20,25,29],[10,15,20,25,29])
plt.xlabel("Matrix size N in log2 scale")
plt.ylabel("Elapse time in log2 scale (millisecond)")
plt.legend()
plt.savefig("bonus2.pdf")
# %%
