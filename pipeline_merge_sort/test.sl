#!/bin/bash
#SBATCH -p qcpu_exp
#SBATCH -A DD-23-135
#SBATCH -nodes 1 
#SBATCH --tasks-per-node=14
#SBATCH -t 1:00:00
#SBATCH --mail-type END
#SBATCH -J TEST-pms

cd $SLURM_SUBMIT_DIR

ml OpenMPI/4.1.4-GCC-11.3.0

mpic++ -O3 -o pms ../pms.cpp
bash test.sh > test.log
