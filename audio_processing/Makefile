MAKEFILE_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CURRENT_DIR = $(shell pwd)
PROJECT_DIR = $(MAKEFILE_DIR:$(CURRENT_DIR)/%=%)

CC = mpic++
CFLAGS = -Wall -Wextra -std=c++20 -O3 -Wno-cast-function-type -fopenmp -lrt -lm -lasound -pthread -lsndfile
CPP_SRC = $(wildcard $(PROJECT_DIR)/src/*.cpp)
CPP_SRC_LIBS = $(CPP_SRC) /usr/local/lib/libportaudio.a 

TARGET = $(PROJECT_DIR)/rtap

.PHONY: all test debug clean

all:
	$(CC) $(CPP_SRC_LIBS) $(CFLAGS) -o $(TARGET)

clean: 
	rm -f $(TARGET)
