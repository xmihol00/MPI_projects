#!/bin/bash

# =======================================================================================================================================================
# Project:         John Conway's Game of Life simulation
# Author:          David Mihola
# E-mail:          xmihol00@stud.fit.vutbr.cz
# Date:            1. 4. 2024
# Description:     Script for compilation and launching of the program. 
# =======================================================================================================================================================

# compile the program
mpic++ --prefix /usr/local/share/OpenMPI -Wall -Wextra -std=c++17 -O3 -Wno-cast-function-type -fopenmp life.cpp -o life

N=4           # number of processes, can be changed if the grid will be evenly divisible by number of processes in each dimension
grid_file=$1  # path to the file with the grid
iterations=$2 # number of iterations
additional_argumnts="${@:3}"  # additional arguments for the program, '-nfp' or '-ep' might be useful for testing

# run the program
mpiexec -n $N ./life $grid_file $iterations $additional_argumnts
