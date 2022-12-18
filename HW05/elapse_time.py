# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
# A = np.array(range(25)).reshape(5,5)
# B = np.array(np.linspace(24,0,25)).reshape(5,5)
A = np.ones([2**8+1, 2**8+1])
B = np.ones([2**8+1, 2**8+1])
C = np.matmul(A,B)
print(C)
# %%
elapsed_times_128 = []
with open("task1_128.txt", "r") as f:
    for i,line in enumerate(f):
        if i%2 == 1:
            elapsed_times_128.append(np.log2(float(line)))

elapsed_times_1024 = []
with open("task1_1024.txt", "r") as f:
    for i,line in enumerate(f):
        if i%2 == 1:
            elapsed_times_1024.append(np.log2(float(line)))
plt.figure()
plt.plot(range(10,31), elapsed_times_128, label="thread:128", marker='*')
plt.plot(range(10,31), elapsed_times_1024, label="thread:1024", marker='*')
plt.xticks([10,15,20,25,29],[10,15,20,25,29])
plt.xlabel("Vector size N in log2 scale")
plt.ylabel("Elapse time in log2 scale (millisecond)")
plt.legend()
plt.savefig("task1.pdf")
# %%
elapsed_time_int = []
elapsed_time_float = []
elapsed_time_double = []

with open("task2_32.txt", "r") as f:
    for i,line in enumerate(f):
        if i % 9 == 2:
            elapsed_time_int.append(np.log2(float(line)))
        if i % 9 == 5:
            elapsed_time_float.append(np.log2(float(line)))
        if i % 9 == 8:
            elapsed_time_double.append(np.log2(float(line)))

plt.figure()
plt.plot(range(5,15), elapsed_time_int, label="int", marker='*')
plt.plot(range(5,15), elapsed_time_float, label="float", marker='*')
plt.plot(range(5,15), elapsed_time_double, label="double", marker='*')
plt.xticks([5,8,10,12,14],[5,8,10,12,14])
plt.xlabel("Vector size N in log2 scale")
plt.ylabel("Elapse time scale (millisecond)")
plt.legend()
# %%
elapsed_time_int = []
elapsed_time_float = []
elapsed_time_double = []

with open("task2_op.txt", "r") as f:
    for i,line in enumerate(f):
        if i % 9 == 2:
            elapsed_time_int.append(np.log2(float(line)))
        if i % 9 == 5:
            elapsed_time_float.append(np.log2(float(line)))
        if i % 9 == 8:
            elapsed_time_double.append(np.log2(float(line)))

plt.figure()
plt.plot(range(2,6), elapsed_time_int, label="int", marker='*')
plt.plot(range(2,6), elapsed_time_float, label="float", marker='*')
plt.plot(range(2,6), elapsed_time_double, label="double", marker='*')
# plt.xticks([5,8,10,12,14],[5,8,10,12,14])
plt.xlabel("block dim in log2 scale")
plt.ylabel("Elapse time in log2 scale (millisecond)")
plt.legend()
# %%
# f and g
elapsed_times_sq = []
with open("matmul.txt", "r") as f:
    for line in f:
        elapsed_times_sq.append(np.log2(float(line)))

elapsed_times_1024 = []
with open("task1_1024 copy.txt", "r") as f:
    for line in f:
        elapsed_times_1024.append(np.log2(float(line)))

elapsed_time_int = []
with open("task2_16.txt", "r") as f:
    for i,line in enumerate(f):
        if i % 9 == 2:
            elapsed_time_int.append(np.log2(float(line)))
plt.figure()
plt.plot(range(5, len(elapsed_times_sq)+5), elapsed_times_sq, label="CPU", marker='*')
plt.plot(range(5, len(elapsed_times_1024)+5), elapsed_times_1024, label="GPU+naive", marker='*')
plt.plot(range(5,15), elapsed_time_int, label="GPU+tile", marker='*')
plt.xticks([5,7,10,12,14],[5,7,10,12,14])
plt.xlabel("Matrix size N in log2 scale")
plt.ylabel("Elapse time in log2 scale (millisecond)")
plt.legend()




# %%
