#!/usr/bin/env bash
#SBATCH --account=DD-23-135
#SBATCH --job-name=PPP_PROJ01_HYBRID_2D
#SBATCH -p qcpu
#SBATCH -t 05:00:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --distribution=block:cyclic:cyclic,NoPack

source load_modules.sh

STDOUT_FILE="run_full_hybrid_2d_out.csv"
STDERR_FILE="run_full_hybrid_2d_err.txt"
BINARY_PATH="../build/ppp_proj01"

# Clear the stdout and stderr files
rm -f $STDOUT_FILE $STDERR_FILE

USER_SCRATCH_PATH=/scratch/project/dd-23-135/$USER

mkdir -p $USER_SCRATCH_PATH

OUT_FILE_PATH=$USER_SCRATCH_PATH/$SLURM_JOBID

mkdir -p $OUT_FILE_PATH

# Doplnte vhodne nastavenie Lustre file system #
################################################
#lfs setstripe -S 1M -c 16 /scratch/project/dd-23-135/$USER
################################################

DISK_WRITE_INTENSITY=50

export KMP_AFFINITY=compact

for procs in 1 4 16 32; do
    for size in 256 512 1024 2048 4096; do
        B="-b"
        
        if [ "$procs" -eq 1 ]; then
            nnodes=1
            n_iters=`expr $((2000000/$size))`
            export OMP_NUM_THREADS=1
            modeP2P=0
            modeRMA=0
            if [ "$size" -eq 256 ]; then
                B="-B"
            fi
        else
            nnodes=$((procs/SLURM_NTASKS_PER_NODE))
            n_iters=`expr $((20000000/$size))`
            export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
            modeP2P=1
            modeRMA=2
        fi

        INPUT=input_data_$size.h5
        OUTPUT=$OUT_FILE_PATH/${size}x${size}_out_hybrid_2d.h5
        
        srun -N $nnodes -n $procs $BINARY_PATH $B -g    -n $n_iters -m $modeP2P -w $DISK_WRITE_INTENSITY -i $INPUT -t $OMP_NUM_THREADS            >> $STDOUT_FILE 2>> $STDERR_FILE
        srun -N $nnodes -n $procs $BINARY_PATH -b -g    -n $n_iters -m $modeP2P -w $DISK_WRITE_INTENSITY -i $INPUT -t $OMP_NUM_THREADS -o $OUTPUT >> $STDOUT_FILE 2>> $STDERR_FILE
        srun -N $nnodes -n $procs $BINARY_PATH -b -g -p -n $n_iters -m $modeP2P -w $DISK_WRITE_INTENSITY -i $INPUT -t $OMP_NUM_THREADS -o $OUTPUT >> $STDOUT_FILE 2>> $STDERR_FILE

        srun -N $nnodes -n $procs $BINARY_PATH -b -g    -n $n_iters -m $modeRMA -w $DISK_WRITE_INTENSITY -i $INPUT -t $OMP_NUM_THREADS            >> $STDOUT_FILE 2>> $STDERR_FILE
        srun -N $nnodes -n $procs $BINARY_PATH -b -g    -n $n_iters -m $modeRMA -w $DISK_WRITE_INTENSITY -i $INPUT -t $OMP_NUM_THREADS -o $OUTPUT >> $STDOUT_FILE 2>> $STDERR_FILE
        srun -N $nnodes -n $procs $BINARY_PATH -b -g -p -n $n_iters -m $modeRMA -w $DISK_WRITE_INTENSITY -i $INPUT -t $OMP_NUM_THREADS -o $OUTPUT >> $STDOUT_FILE 2>> $STDERR_FILE

        rm -f $OUTPUT
    done
done

rm -rf $OUT_FILE_PATH
