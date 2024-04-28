MAKEFILE_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CURRENT_DIR = $(shell pwd)
PROJECT_DIR = $(MAKEFILE_DIR:$(CURRENT_DIR)/%=%)

CC = mpic++
CFLAGS = -Wall -Wextra -std=c++20 -O3 -Wno-cast-function-type -lrt -lm -lasound -pthread -lsndfile
CPP_SRC = $(wildcard $(PROJECT_DIR)/*.cpp)
CPP_SRC_LIBS = $(CPP_SRC) /usr/local/lib/libportaudio.a 

TARGET = $(PROJECT_DIR)/rtap
TARGET_TEST = $(PROJECT_DIR)/rtap_test
TARGET_DEBUG = $(PROJECT_DIR)/rtap_debug

.PHONY: all test debug clean

all:
	$(CC) $(CPP_SRC_LIBS) $(CFLAGS) -o $(TARGET)

test:
	$(CC) $(CPP_SRC_LIBS) $(CFLAGS) -D_TEST_PRINT_ -o $(TARGET_TEST)

debug:
	$(CC) $(CPP_SRC_LIBS) $(CFLAGS) -D_DEBUG_PRINT_ -o $(TARGET_DEBUG)

clean: 
	rm -f $(TARGET) $(TARGET_TEST)
