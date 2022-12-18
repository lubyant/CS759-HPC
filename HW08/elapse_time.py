# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
with open("task1.txt", "r") as f:
    elapse_time = []
    for i, line in enumerate(f):
        if i % 3 == 2:
            elapse_time.append(float(line))


plt.figure()
plt.plot(range(1, 21), elapse_time, marker="*")
plt.xticks([1, 10, 20], [1, 10, 20])
plt.xlabel("Number of scores t ")
plt.ylabel("Elapse time in (millisecond)")
plt.title("Task1 - Matrix Multiplication")
plt.savefig("task1.pdf")

# %%
elapsed_time = []
with open("task2.txt", "r") as f:
    for i,line in enumerate(f):
        if i % 3 == 2:
            elapsed_time.append(float(line))

plt.figure()
plt.plot(range(1, 21), elapsed_time, marker="*")
plt.xticks([1, 10, 20], [1, 10, 20])
plt.xlabel("Number of scores t ")
plt.ylabel("Elapse time in (millisecond)")
plt.title("Task2 - Convolution")
plt.savefig("task2.pdf")

# %%
elapsed_time = []
with open("task3_ts.txt", "r") as f:
    for i,line in enumerate(f):
        if i % 3 == 2:
            elapsed_time.append(float(line))

plt.figure()
plt.plot(range(1, 11), elapsed_time, marker="*")
plt.xticks([1, 5, 10], [1, 5, 10])
plt.xlabel("threshold size in log2 scale")
plt.ylabel("Elapse time in log2 scale millisecond")
plt.title("Task3 - merge sort at different thres")
plt.savefig("task3_ts.pdf")
# %%
elapsed_time = []
with open("task3_t256.txt", "r") as f:
    for i,line in enumerate(f):
        if i % 3 == 2:
            elapsed_time.append(float(line))

plt.figure()
plt.plot(range(1, 21), elapsed_time, marker="*")
plt.xticks([1, 10, 20], [1, 10, 20])
plt.xlabel("Cores number")
plt.ylabel("Elapse time in millisecond")
plt.title("Task3 - merge sort threshold = 256")
plt.savefig("task3_t.pdf")
# %%
