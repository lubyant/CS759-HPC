# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
with open("task1_fs.txt", "r") as f:
    elapse_time_fs = []
    for i, line in enumerate(f):
        if i % 3 == 2:
            elapse_time_fs.append(float(line))

with open("task1_red.txt", "r") as f:
    elapse_time_red = []
    for i, line in enumerate(f):
        if i % 3 == 2:
            elapse_time_red.append(float(line))

plt.figure()
plt.plot(range(1, 11), elapse_time_fs, marker="*", label="false sharing")
plt.plot(range(1, 11), elapse_time_red, marker="*", label="none false sharing")
plt.xticks([1, 5, 10], [1, 5, 10])
plt.xlabel("Number of scores t ")
plt.ylabel("Elapse time in (millisecond)")
plt.title("Task1 - distance")
plt.legend()
plt.savefig("task1.pdf")

# %%
elapsed_time_nosimd = []
with open("task2.txt", "r") as f:
    for i,line in enumerate(f):
        if i % 2 == 1:
            elapsed_time_nosimd.append(float(line))

elapsed_time_simd = []
with open("task2_simd.txt", "r") as f:
    for i,line in enumerate(f):
        if i % 2 == 1:
            elapsed_time_simd.append(float(line))

plt.figure()
plt.plot(range(1, 11), elapsed_time_nosimd, marker="*", label="no simd")
plt.plot(range(1, 11), elapsed_time_simd, marker="*", label="simd")
plt.xticks([1, 5, 10], [1, 5, 10])
plt.xlabel("Number of scores t ")
plt.ylabel("Elapse time in (millisecond)")
plt.title("Task2 - montecarlo")
plt.legend()
plt.savefig("task2.pdf")


# %%
elapsed_time = []
with open("task3.txt", "r") as f:
    for i,line in enumerate(f):
        elapsed_time.append(np.log2(float(line)))

plt.figure()
plt.plot(range(1, 26), elapsed_time, marker="*")
plt.xticks([1, 10, 25], [1, 10, 25])
plt.xlabel("Array length in log scale")
plt.ylabel("Elapse time in log scale of millisecond")
plt.title("Task3 - MPI")
plt.savefig("task3.pdf")
# %%
