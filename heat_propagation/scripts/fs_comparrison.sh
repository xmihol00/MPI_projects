#!/usr/bin/env bash
#SBATCH --account=DD-23-135
#SBATCH --job-name=FS_COMPARRISON
#SBATCH -p qcpu
#SBATCH -t 05:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=32
#SBATCH --distribution=block:block:block,Pack

source load_modules.sh

declare -a SIZES=(1024 2048 4096)
declare -a PROCESSES=(4 16 64)
declare -a NODES=(1 1 2)
declare -a STRIPES=("1K" "2K" "4K" "8K" "16K" "32K" "64K" "128K" "256K" "512K" "1M" "2M")

STDOUT_FILE="fs_comparrison_out.csv"
STDERR_FILE="fs_comparrison_err.txt"
BINARY_PATH="../build/ppp_proj01"

# Clear the stdout and stderr files
rm -f $STDOUT_FILE $STDERR_FILE

USER_SCRATCH_PATH=/scratch/project/dd-23-135/$USER
OUT_FILE_PATH=$USER_SCRATCH_PATH/$SLURM_JOBID

mkdir -p $USER_SCRATCH_PATH
mkdir -p $OUT_FILE_PATH

DISK_WRITE_INTENSITY=25
echo "output_type;stripe;mpi_procs;grid_tiles_x;grid_tiles_y;omp_threads;domain_size;n_iterations;disk_write_intensity;airflow;material_file;output_file;simulation_mode;middle_col_avg_temp;total_time;iteration_time" > $STDOUT_FILE

for stripe in ${STRIPES[*]}; do
    OUT_DIR=$OUT_FILE_PATH/stripe_${stripe}
    mkdir -p $OUT_DIR
    echo "Executing: lfs setstripe -S $stripe -c 16 $OUT_DIR"
    lfs setstripe -S $stripe -c 16 $OUT_DIR

    for i in 0 1 2; do
        procs=${PROCESSES[$i]}
        nnodes=${NODES[$i]}
        size=${SIZES[$i]}

        INPUT=input_data_$size.h5
        OUTPUT=$OUT_DIR/${size}x${size}_out.h5

        echo "Executing: srun -N $nnodes -n $procs $BINARY_PATH -b -g -n 2505 -m 2 -w $DISK_WRITE_INTENSITY -i $INPUT -o $OUTPUT"
        echo -n "seq;$stripe;" >> $STDOUT_FILE
        srun -N $nnodes -n $procs $BINARY_PATH -b -g -n 2505 -m 2 -w $DISK_WRITE_INTENSITY -i $INPUT -o $OUTPUT >> $STDOUT_FILE 2>> $STDERR_FILE

        echo "Executing: srun -N $nnodes -n $procs $BINARY_PATH -b -g -p -n 2505 -m 2 -w $DISK_WRITE_INTENSITY -i $INPUT -o $OUTPUT"
        echo -n "par;$stripe;" >> $STDOUT_FILE
        srun -N $nnodes -n $procs $BINARY_PATH -b -g -p -n 2505 -m 2 -w $DISK_WRITE_INTENSITY -i $INPUT -o $OUTPUT >> $STDOUT_FILE 2>> $STDERR_FILE
        rm -f $OUTPUT
    done
done

rm -rf $OUT_FILE_PATH
