MAKEFILE_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CURRENT_DIR = $(shell pwd)
PROJECT_DIR = $(MAKEFILE_DIR:$(CURRENT_DIR)/%=%)

CC = mpic++
CFLAGS = -Wall -Wextra -std=c++20 -O3 -Wno-cast-function-type -fopenmp -lrt -lm -lasound -pthread -lsndfile
CPP_SRC = $(wildcard $(PROJECT_DIR)/src/*.cpp)
CPP_SRC_LIBS = $(CPP_SRC) /usr/local/lib/libportaudio.a 

TARGET = $(PROJECT_DIR)/rtap
PACK_TARGET = $(PROJECT_DIR)/RTSa-project-outcome-xmihol00.zip

.PHONY: all clean pack

all:
	$(CC) $(CPP_SRC_LIBS) $(CFLAGS) -o $(TARGET)

clean: 
	rm -f $(TARGET) $(PACK_TARGET)

pack: clean
	zip -r $(PACK_TARGET) ./src ./Makefile ./README.md ./mpi_echo ./recorder ./processed_files/ ./run_simulations.sh ./test_recording.wav ./report.pdf -x "processed_files/*wav"