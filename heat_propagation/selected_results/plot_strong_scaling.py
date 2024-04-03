import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

SMALL_SIZE = 13
LARGE_SIZE = 15

matplotlib.rc('font', size=SMALL_SIZE)        
matplotlib.rc('axes', titlesize=SMALL_SIZE)   
matplotlib.rc('axes', labelsize=SMALL_SIZE)  
matplotlib.rc('xtick', labelsize=SMALL_SIZE)  
matplotlib.rc('ytick', labelsize=SMALL_SIZE)  
matplotlib.rc('legend', fontsize=SMALL_SIZE)  
matplotlib.rc('figure', titlesize=LARGE_SIZE) 

df = pd.read_csv("strong_scaling.csv", sep=";")
df = df[["mpi_procs", "domain_size", "iteration_time"]]
df_256_domain = df[df["domain_size"] == 256].copy()
df_256_domain.reset_index(drop=True, inplace=True)
df_4096_domain = df[df["domain_size"] == 4096].copy()
df_4096_domain.reset_index(drop=True, inplace=True)

df_256_domain.loc[:, "iteration_time"] /= df_256_domain.loc[0, "iteration_time"]
df_4096_domain.loc[:, "iteration_time"] /= df_4096_domain.loc[0, "iteration_time"]

processors = df_256_domain["mpi_procs"].to_numpy()
ideal_times = np.ones_like(processors)
ideal_times = ideal_times / processors

plt.figure(figsize=(6, 6))
plt.title("Strong scaling", {"fontsize": LARGE_SIZE})
plt.plot(processors, df_256_domain["iteration_time"], label="MPI 2D 256x256 RMA", marker="D", markersize=10)
plt.plot(processors, df_4096_domain["iteration_time"], label="MPI 2D 4096x4096 RMA", marker="s", markersize=10)
plt.plot(processors, ideal_times, label="Perfect scaling", linestyle="--", marker="o", markersize=10)
plt.xlabel("Number of processors")
plt.xscale("log", base=2)
plt.ylabel("Normalized iteration time")
plt.yscale("log", base=2)
plt.legend()
plt.grid()
plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.savefig("strong_scaling.png", dpi=500)