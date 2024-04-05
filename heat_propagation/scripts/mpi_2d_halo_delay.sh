#!/usr/bin/env bash
#SBATCH --account=DD-23-135
#SBATCH --job-name=HALO_DELAY
#SBATCH -p qcpu
#SBATCH -t 05:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=32
#SBATCH --distribution=block:block:block,Pack

source load_modules.sh

declare -a SIZES=(256 512 1024 2048 4096)
declare -a PROCESSES=(16 32 64 128)

STDOUT_FILE="mpi_2d_halo_delay_out.csv"
STDERR_FILE="mpi_2d_halo_delay_err.txt"
rm -f $STDOUT_FILE $STDERR_FILE

BINARY_PATH="../build/ppp_proj01"

for procs in ${PROCESSES[*]}; do
    for size in ${SIZES[*]}; do
        nnodes=$((procs/SLURM_NTASKS_PER_NODE))
        if [ "$nnodes" -eq 0 ]; then
            nnodes=1
        fi

        n_iters=`expr $((20000000/$size))`
        modeP2P=1
        modeRMA=2

        INPUT=input_data_$size.h5
        
        srun -N $nnodes -n $procs $BINARY_PATH -b -g -n $n_iters -m $modeP2P -i $INPUT >> $STDOUT_FILE 2>> $STDERR_FILE; echo ";$modeP2P" >> $STDOUT_FILE
        srun -N $nnodes -n $procs $BINARY_PATH -b -g -n $n_iters -m $modeRMA -i $INPUT >> $STDOUT_FILE 2>> $STDERR_FILE; echo ";$modeRMA" >> $STDOUT_FILE
    done
done
