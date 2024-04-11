MAKEFILE_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CURRENT_DIR = $(shell pwd)
PROJECT_DIR = $(MAKEFILE_DIR:$(CURRENT_DIR)/%=%)

CC = mpic++
CFLAGS = -Wall -Wextra -std=c++20 -fopenmp -O3 -Wno-cast-function-type
CPP_SRC = $(wildcard $(PROJECT_DIR)/*.cpp)

TARGET = $(PROJECT_DIR)/rtap
TARGET_TEST = $(PROJECT_DIR)/rtap_test
TARGET_DEBUG = $(PROJECT_DIR)/rtap_debug

.PHONY: all test debug clean

all:
	$(CC) $(CFLAGS) $(CPP_SRC) -o $(TARGET)

test:
	$(CC) $(CFLAGS) $(CPP_SRC) -D_TEST_PRINT_ -o $(TARGET_TEST)

debug:
	$(CC) $(CFLAGS) $(CPP_SRC) -D_DEBUG_PRINT_ -o $(TARGET_DEBUG)

clean: 
	rm -f $(TARGET) $(TARGET_TEST)
