# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
with open("task1.txt", "r") as f:
    elapse_time = []
    for line in f:
        elapse_time.append(np.log2(float(line)))

plt.figure()
plt.plot(range(5,16), elapse_time, marker="*")
plt.xticks([5, 10, 15], [5, 10, 15])
plt.xlabel("Matrix size N in log2 scale")
plt.ylabel("Elapse time in log2 scale (millisecond)")
plt.title("Average overt 20 times")
plt.savefig("task1.pdf")

elapsed_time_int = []
with open("task2_16.txt", "r") as f:
    for i,line in enumerate(f):
        if i % 9 == 2:
            elapsed_time_int.append(np.log2(float(line)))

plt.figure()
plt.plot(range(5,16), elapse_time, marker="*", label="cublas")
plt.plot(range(5,15), elapsed_time_int, label="GPU+tile", marker='*')
plt.xticks([5, 10, 15], [5, 10, 15])
plt.legend()
plt.xlabel("Matrix size N in log2 scale")
plt.ylabel("Elapse time in log2 scale (millisecond)")

# %%
elapse_time = []
with open("task2.txt", "r") as f:
    for i, line in enumerate(f):
        if i%2 == 1:
            elapse_time.append(np.log2(float(line)))

plt.figure()
plt.plot(range(10,21), elapse_time, marker="*")
plt.xticks([10, 15, 20], [10, 15, 20])
plt.xlabel("Vector size N in log2 scale")
plt.ylabel("Elapse time in log2 scale (millisecond)")
plt.title("Elapse time for inclusive scan algorithm")
plt.savefig("task2.pdf")
# %%
