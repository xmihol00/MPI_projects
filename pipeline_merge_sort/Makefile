MAKEFILE_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CURRENT_DIR = $(shell pwd)
PROJECT_DIR = $(MAKEFILE_DIR:$(CURRENT_DIR)/%=%)

CC = mpic++
CFLAGS = -Wall -Wextra -std=c++17 -g
CPP_SRC = $(wildcard $(PROJECT_DIR)/*.cpp)

ifndef N  # by default run 3 processes, i.e. 2^2=4 values
    N := 3
endif
M = $(shell (echo "(2^($N-1))" | bc -l | xargs printf "%1.0f")) # compute 2^(N-1)

.PHONY: all run clean

all: $(PROJECT_DIR)/pms.cpp
	$(CC) $(CFLAGS) $(CPP_SRC) -o $(PROJECT_DIR)/pms

run: all
	dd if=/dev/random bs=1 count=$M 2>/dev/null | mpirun --oversubscribe -np $N $(PROJECT_DIR)/pms $D | python3 $(PROJECT_DIR)/sorted.py $D

# usage: 'N' stands for number of processes and 'D' stands for direction ('-a' for ascending, '-d' for descending) 
#        make
#        make run
#        make -f some/complicated/path/pipeline_merge_sort/Makefile
#        make run N=4 D=-a
#        make run N=6 D=-d
#        make -f pipeline_merge_sort/Makefile run N=6 D=-a
#        make -f pipeline_merge_sort/Makefile run N=10 D=-d
#        make -f some/complicated/path/pipeline_merge_sort/Makefile run N=10