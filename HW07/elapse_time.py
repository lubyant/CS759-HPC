# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
with open("task1_thrust.txt", "r") as f:
    elapse_time_thrust = []
    for i, line in enumerate(f):
        if i % 2 != 0:
            elapse_time_thrust.append(np.log2(float(line)))

with open("task1_cub.txt", "r") as f:
    elapse_time_cub = []
    for i, line in enumerate(f):
        if i % 2 != 0:
            elapse_time_cub.append(np.log2(float(line)))

with open("task1_128.txt", "r") as f:
    elapse_time_gpu = []
    for i, line in enumerate(f):
        if i % 2 != 0:
            elapse_time_gpu.append(np.log2(float(line)))

plt.figure()
plt.plot(range(10,31), elapse_time_thrust, marker="*", label="thrust")
plt.plot(range(10,31), elapse_time_cub, marker="*", label="cub")
plt.plot(range(10,31), elapse_time_gpu, marker="*", label="gpu")
plt.xticks([10, 15, 30], [10, 15, 30])
plt.xlabel("Reduction size N in log2 scale")
plt.ylabel("Elapse time in log2 scale (millisecond)")
plt.title("Task1 - Array reduction")
plt.legend()
plt.savefig("task1.pdf")

# %%
elapsed_time = []
with open("task2.txt", "r") as f:
    for i,line in enumerate(f):
        if i % 3 == 2:
            elapsed_time.append(np.log2(float(line)))

plt.figure()
plt.plot(range(5,25), elapsed_time, marker="*")
plt.xticks([5, 15, 24], [5, 15, 24])
plt.legend()
plt.xlabel("Count array size N in log2 scale")
plt.ylabel("Elapse time in log2 scale (millisecond)")
plt.title("Task2 - count")
plt.savefig("task2.pdf")

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
