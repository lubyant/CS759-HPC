# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
with open("task11.txt", "r") as f:
    elapse_time = []
    for i, line in enumerate(f):
        if i % 2 == 1:
            elapse_time.append(float(line))

plt.figure()
plt.bar([1,2,3,4,5,6], elapse_time)

plt.xlabel("Optimize X ")
plt.ylabel("Elapse time in (millisecond)")
plt.title("Task11")
plt.legend()
plt.savefig("task11.pdf")


with open("task12.txt", "r") as f:
    elapse_time = []
    for i, line in enumerate(f):
        if i % 2 == 1:
            elapse_time.append(float(line))

plt.figure()
plt.bar([1,2,3,4,5,6], elapse_time)

plt.xlabel("Optimize X ")
plt.ylabel("Elapse time in (millisecond)")
plt.title("Task12")
plt.legend()
plt.savefig("task12.pdf")

with open("task13.txt", "r") as f:
    elapse_time = []
    for i, line in enumerate(f):
        if i % 2 == 1:
            elapse_time.append(float(line))

plt.figure()
plt.bar([1,2,3,4,5,6], elapse_time)

plt.xlabel("Optimize X ")
plt.ylabel("Elapse time in (millisecond)")
plt.title("Task13")
plt.legend()
plt.savefig("task13.pdf")

with open("task14.txt", "r") as f:
    elapse_time = []
    for i, line in enumerate(f):
        if i % 2 == 1:
            elapse_time.append(float(line))

plt.figure()
plt.bar([1,2,3,4,5,6], elapse_time)

plt.xlabel("Optimize X ")
plt.ylabel("Elapse time in (millisecond)")
plt.title("Task14")
plt.legend()
plt.savefig("task14.pdf")

# %%
elapsed_time_1 = []
elapsed_time_2 = []
with open("task2_pure.txt", "r") as f:
    for i,line in enumerate(f):
        if i % 4 == 1:
            elapsed_time_1.append(float(line))
        if i % 4 == 3:
            elapsed_time_2.append(float(line))



plt.figure()
plt.plot(range(1, 21), elapsed_time_1, marker="*", label="MPI + OMP")
plt.plot(range(1, 21), elapsed_time_2, marker="*", label="pure OMP")
plt.xticks([1, 10, 20], [1, 10, 20])
plt.xlabel("Number of threads t ")
plt.ylabel("Elapse time in (millisecond)")
plt.title("Task2 - pure")
plt.legend()
plt.savefig("task2.pdf")


# %%
elapsed_time_1 = []
elapsed_time_2 = []
with open("task2_comp.txt", "r") as f:
    for i,line in enumerate(f):
        if i % 4 == 1:
            elapsed_time_1.append(np.log2(float(line)))
        if i % 4 == 3:
            elapsed_time_2.append(np.log2(float(line)))

plt.figure()
plt.plot(range(1, 27), elapsed_time_1, marker="*", label="MPI + OMP")
plt.plot(range(1, 27), elapsed_time_2, marker="*", label="pure OMP")
plt.xticks([1, 10, 25], [1, 10, 25])
plt.xlabel("Array length in log2 scale")
plt.ylabel("Elapse time in log2 scale of millisecond")
plt.title("Task2 - comparision")
plt.legend()
plt.savefig("task2_comp.pdf")
# %%
