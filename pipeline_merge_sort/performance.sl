#!/bin/bash
#SBATCH -p qcpu_exp
#SBATCH -A DD-23-135
#SBATCH -nodes 1 
#SBATCH -t 1:00:00
#SBATCH --ntasks-per-node 32
#SBATCH --mail-type END
#SBATCH -J PERF-pms

cd $SLURM_SUBMIT_DIR

ml OpenMPI/4.1.4-GCC-11.3.0

mpic++ -O3 -o pms ../pms.cpp
bash performance.sh >> performance.log
