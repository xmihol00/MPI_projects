import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

SMALL_SIZE = 12.5
LARGE_SIZE = 15

matplotlib.rc('font', size=SMALL_SIZE)        
matplotlib.rc('axes', titlesize=SMALL_SIZE)   
matplotlib.rc('axes', labelsize=SMALL_SIZE)  
matplotlib.rc('xtick', labelsize=SMALL_SIZE)  
matplotlib.rc('ytick', labelsize=SMALL_SIZE)  
matplotlib.rc('legend', fontsize=SMALL_SIZE)  
matplotlib.rc('figure', titlesize=LARGE_SIZE) 

df = pd.read_csv("weak_scaling.csv", sep=";")
df = df[["decomposition", "mpi_procs", "omp_threads", "iteration_time"]]
df["mpi_procs"] *= df["omp_threads"]
df_mpi_2d = df[df["decomposition"] == "2D"].copy()
df_mpi_2d.reset_index(drop=True, inplace=True)
df_hybrid_1d = df[df["decomposition"] == "1D"].copy()
df_hybrid_1d.reset_index(drop=True, inplace=True)

all_processors = df["mpi_procs"].unique()
all_processors.sort()

ffs = lambda x: int(x & (-x)).bit_length() - 1

# adjustment for the number of processors, i.e. 256x256 domain is computed by a single process, while already 512x512 domain is computed by 16 or 32 processes
mpi_2d_powers = np.array([ffs(p) for p in df_mpi_2d["mpi_procs"]])
hybrid_1d_powers = np.array([ffs(p) for p in df_hybrid_1d["mpi_procs"]])
mpi_2d_powers_diffs = np.array([0] + [mpi_2d_powers[i] - mpi_2d_powers[i - 1] - 1 for i in range(1, len(mpi_2d_powers))])
hybrid_1d_powers_diffs = np.array([0] + [hybrid_1d_powers[i] - hybrid_1d_powers[i - 1] - 1 for i in range(1, len(hybrid_1d_powers))])
mpi_2d_powers_diffs_prescan = np.cumsum(mpi_2d_powers_diffs)
hybrid_1d_powers_diffs_prescan = np.cumsum(hybrid_1d_powers_diffs)

# normalization by the quadratic computation complexity based on domain size
mpi_2d_computation_complexity = np.linspace(0, len(mpi_2d_powers) - 1, len(mpi_2d_powers))
hybrid_1d_computation_complexity = np.linspace(0, len(hybrid_1d_powers) - 1, len(hybrid_1d_powers))

mpi_2d_normalization_factor = (2 ** mpi_2d_computation_complexity) / (2 ** mpi_2d_powers_diffs_prescan)
hybrid_1d_normalization_factor = (2 ** hybrid_1d_computation_complexity) / (2 ** hybrid_1d_powers_diffs_prescan)
df_mpi_2d.loc[:, "iteration_time"] /= df_mpi_2d.loc[0, "iteration_time"] * mpi_2d_normalization_factor
df_hybrid_1d.loc[:, "iteration_time"] /= df_hybrid_1d.loc[0, "iteration_time"] * hybrid_1d_normalization_factor

ideal_times = np.ones_like(all_processors)

plt.figure(figsize=(6, 6))
plt.title("Weak scaling", {"fontsize": LARGE_SIZE})
plt.plot(all_processors, ideal_times, label="Perfect scaling")
plt.plot(df_hybrid_1d["mpi_procs"], df_hybrid_1d["iteration_time"], label="Hybrid 1D RMA", marker="s", markersize=8, linestyle="--")
plt.plot(df_mpi_2d["mpi_procs"], df_mpi_2d["iteration_time"], label="MPI 2D RMA", marker="D", markersize=8, linestyle="--")
plt.xlabel("Number of processes")
plt.xscale("log", base=2)
plt.ylim(1 / 2**4.2, 2**4.2)
plt.ylabel("Normalized iteration time")
plt.yscale("log", base=2)
plt.legend()
plt.grid()
plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.savefig("weak_scaling.png", dpi=500)