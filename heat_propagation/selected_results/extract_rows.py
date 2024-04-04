import pandas as pd 

#../results_with_fs_raw/run_full_mpi_2d_out.csv
#../results_with_fs_data_type/run_full_mpi_2d_out.csv
#../results_no_fs_raw/run_full_mpi_2d_out.csv
#../results_no_fs_data_type/run_full_mpi_2d_out.csv

df1 = pd.read_csv("../results_with_fs_raw/run_full_mpi_2d_out.csv", sep=";")
df1 = df1.loc[[7, 37, 67, 97, 127], :]
df1["run"] = 0
df1["fs"] = 1

df2 = pd.read_csv("../results_with_fs_data_type/run_full_mpi_2d_out.csv", sep=";")
df2 = df2.loc[[7, 37, 67, 97, 127], :]
df2["run"] = 1
df2["fs"] = 1

df3 = pd.read_csv("../results_no_fs_raw/run_full_mpi_2d_out.csv", sep=";")
df3 = df3.loc[[7, 37, 67, 97, 127], :]
df3["run"] = 0
df3["fs"] = 0

df4 = pd.read_csv("../results_no_fs_data_type/run_full_mpi_2d_out.csv", sep=";")
df4 = df4.loc[[7, 37, 67, 97, 127], :]
df4["run"] = 1
df4["fs"] = 0

merged = pd.concat([df1, df2, df3, df4])
merged.to_csv("fs_settings.csv", sep=";", index=False)
