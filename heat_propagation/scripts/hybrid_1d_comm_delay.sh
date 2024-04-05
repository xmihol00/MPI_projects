#!/usr/bin/env bash
#SBATCH --account=DD-23-135
#SBATCH --job-name=COMM_DELAY
#SBATCH -p qcpu
#SBATCH -t 05:00:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --distribution=block:cyclic:cyclic,NoPack

source load_modules.sh

declare -a SIZES=(256 512 1024 2048 4096)
declare -a PROCESSES=(2 4 8 16)

STDOUT_FILE="hybrid_1d_comm_delay_out.csv"
STDERR_FILE="hybrid_1d_comm_delay_err.txt"
rm -f $STDOUT_FILE $STDERR_FILE

BINARY_PATH="../build/ppp_proj01"

export KMP_AFFINITY=compact

for procs in ${PROCESSES[*]}; do
    for size in ${SIZES[*]}; do
        nnodes=$((procs/SLURM_NTASKS_PER_NODE))
        n_iters=`expr $((20000000/$size))`
        export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
        modeP2P=1
        modeRMA=2

        INPUT=input_data_$size.h5
        
        srun -N $nnodes -n $procs $BINARY_PATH -b -n $n_iters -m $modeP2P -i $INPUT -t $OMP_NUM_THREADS >> $STDOUT_FILE 2>> $STDERR_FILE; echo ";$modeP2P" >> $STDOUT_FILE
        srun -N $nnodes -n $procs $BINARY_PATH -b -n $n_iters -m $modeRMA -i $INPUT -t $OMP_NUM_THREADS >> $STDOUT_FILE 2>> $STDERR_FILE; echo ";$modeRMA" >> $STDOUT_FILE        
    done
done
