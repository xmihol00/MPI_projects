MAKEFILE_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CURRENT_DIR = $(shell pwd)
PROJECT_DIR = $(MAKEFILE_DIR:$(CURRENT_DIR)/%=%)

CC = mpic++
CFLAGS = -Wall -Wextra -std=c++17 -O3 -Wno-cast-function-type
CPP_SRC = $(wildcard $(PROJECT_DIR)/*.cpp)

.PHONY: all run clean

all:
	$(CC) $(CFLAGS) $(CPP_SRC) -o $(PROJECT_DIR)/life
