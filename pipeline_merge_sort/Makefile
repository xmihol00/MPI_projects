MAKEFILE_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CURRENT_DIR = $(shell pwd)
PROJECT_DIR = $(MAKEFILE_DIR:$(CURRENT_DIR)/%=%)

CC = mpic++
CFLAGS = -Wall -Wextra -std=c++17 -O3 -Wno-cast-function-type
CPP_SRC = $(wildcard $(PROJECT_DIR)/*.cpp)

ifndef D
	D := -a
endif
ifndef C
	C := -b
endif
ifndef M  # by default run 4 processes, i.e. 2^3=8 values
    M := 8
endif
N=$(shell (python3 -c "from math import ceil, log2; print(ceil(log2($M)+1), end='')"))

.PHONY: all run clean

all: $(PROJECT_DIR)/pms.cpp
	$(CC) $(CFLAGS) $(CPP_SRC) -o $(PROJECT_DIR)/pms

run: all
	dd if=/dev/random bs=1 count=$M 2>/dev/null | mpirun --oversubscribe -np $N $(PROJECT_DIR)/pms $D $C | python3 $(PROJECT_DIR)/sorted.py $D $M

# usage: 'M' stands for number of values to be sorted and 'D' stands for direction ('-a' for ascending, '-d' for descending) 
#        make
#        make run
#        make -f some/complicated/path/pipeline_merge_sort/Makefile
#        make run M=1234 D=-a C=-b
#        make run M=65   D=-d C=-s
#        make -f pipeline_merge_sort/Makefile run M=100  D=-d C=-b
#        make -f pipeline_merge_sort/Makefile run M=1000 D=-a C=-s
#        make -f some/complicated/path/pipeline_merge_sort/Makefile run
