# %%
import numpy as np
import matplotlib.pyplot as plt
# %%
Lx = 1
Ly = 1
nx = 32
ny = 32
imin = 1
imax = imin + nx - 1
jmin = 1
jmax = jmin + ny - 1
data_seq = []
data_omp = []
data_cu = []
x = np.zeros([1, nx+2])
x[0, imin:imax+1+1] = np.linspace(0, Lx, nx+1)
y = np.zeros([1, ny+2])
y[0, jmin:jmax+1+1] = np.linspace(0, Ly, ny+1)
with open("u_seq.txt", "r") as f:
    for line in f:
        for num in line.split(" "):
            try:
                data_seq.append(float(num))
            except:
                pass

with open("u_omp.txt", "r") as f:
    for line in f:
        for num in line.split(" "):
            try:
                data_omp.append(float(num))
            except:
                pass

with open("u_cuda.txt", "r") as f:
    for line in f:
        for num in line.split(" "):
            try:
                data_cu.append(float(num))
            except:
                pass
u = np.array(data_seq)
u = u.reshape(nx+2, nx+2)

u_omp = np.array(data_omp)
u_omp = u_omp.reshape(nx+2, nx+2)

u_cu = np.array(data_omp)
u_cu= u_cu.reshape(nx+2, nx+2)
# %% plotting
plt.figure(figsize=(8, 6), dpi=80)
X, Y = np.meshgrid(x[0, imin:imax+1], y[0, jmin:jmax+1])
plt.contourf(X, Y, u[imin:imax+1, jmin:jmax+1].T, levels=20)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.savefig("results")


plt.figure()

u_c = u[15, :]

plt.plot(y.flatten(), u_c.flatten(), label='Single thread sequential', marker="*")

u_c = u_omp[15, :]

plt.plot(y.flatten(), u_c.flatten(), label='10 thread omp', marker="o")

u_c = u_cu[15, :]

plt.plot(y.flatten(), u_c.flatten(), label='16 block_dim cuda')

ref = np.array([1, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, -
               0.20581, -0.21090, -0.15662, -0.1015, -0.06434, -0.04775, -0.04192, -0.03717, 0])
grid = np.array([129, 126, 125, 124, 123, 110, 95, 80,
                65, 59, 37, 23, 14, 10, 9, 8, 1])/129
plt.scatter(grid, ref, label='Ghia(1982) results', color='black')
plt.xlabel('y/L')
plt.ylabel('u')
plt.savefig("validation")
plt.legend()
plt.show()
# %%
# comparison between seq vs cuda vs omp
seq = []
omp = []
cu = []
with open("size_comp.txt", "r") as f:
    for i, line in enumerate(f):
        if i % 9 == 2:
            seq.append(np.log2(float(line.split(" ")[2])))
        if i % 9 == 5:
            omp.append(np.log2(float(line.split(" ")[2])))
        if i % 9 == 8:
            cu.append(np.log2(float(line.split(" ")[2])))

plt.figure()
plt.plot(range(16, 16+45), seq, label="no parallel", marker="*")
plt.plot(range(16, 16+45), omp, label="omp", marker="+")
plt.plot(range(16, 16+45), cu, label="cuda", marker="o")
plt.legend()
plt.xlabel("Size of CFD field (number of grids)")
plt.ylabel("Elapsed time in log2 scale (millisecond)")
plt.savefig("size_comparison")

# %%
# scale analysis for cuda and omp
cu_time = []
with open("cu_threads.txt", "r") as f:
    for i, line in enumerate(f):
        if i % 3 == 2:
            cu_time.append(np.log2(float(line.split(" ")[2])))

plt.figure()

omp_time = []
with open("omp_threads.txt", "r") as f:
    for i, line in enumerate(f):
        if i % 3 == 2:
            omp_time.append(np.log2(float(line.split(" ")[2])))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
ax1.plot(range(4, len(cu_time)+4), cu_time)
ax1.set_ylabel("Elapsed time in log2 scale (millisecond)")
ax1.set_xlabel("Thread block dimension in log2 scale")
ax1.set_yticks([17.90, 18.00, 18.10])
ax1.set_xticks([4, 5, 6, 7])
ax1.set_title("Scaling analysis for cuda")

ax2.plot(range(1, len(omp_time)+1), omp_time)
ax2.set_ylabel("Elapsed time in log2 scale (millisecond)")
ax2.set_xlabel("Number of the threads")
ax2.set_yticks([15, 16, 17])
ax2.set_xticks(range(1,21))
ax2.set_title("Scaling analysis for OpenMP")

plt.savefig("scaling_analysis")
# %%
