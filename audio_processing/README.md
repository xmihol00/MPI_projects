# Real Time Audio Processing
Framework for real time audio processing and simulation.

## Directory Structure
```
├── processed_files/        - files generated from test_recording.wav with simulation of real time audio processing
├── recorder/               - simple application for audio recording to a WAV file
├── mpi_echo/               - simple echo of microphone input to speaker with MPI processes
├── src/                    - source files of the framework
├── Makefile                - file for building the framework with GNU Make
├── README.md               - this file
├── run_simulations.sh      - script for running the simulation of real time audio processing
└── test_recording.wav      - test audio recording used for simulation and testing
```

## Requirements
The following libraries must be installed:
  * https://www.open-mpi.org,
  * https://www.portaudio.com,
  * https://libsndfile.github.io/libsndfile/.

## Usage
Build the framework by executing the `make` command in the root directory and launch it with the following command:
```
mpiexec -np 2 ./rtap [-s <sampling rate>] [-n <samples per chunk>] [-m <milliseconds per chunk>]
                     [-c <channels>] [-i <input file>] [-o <output file>] [-b <number of buffers>]
                     [-d <processing delay>] [-j <jitter>] [-x] [-h]
Options:
  -s <unsigned integer>:  The sampling rate of the audio data in Hz (default: 22050).
  -n <unsigned integer>:  The number of samples per chunk (window size) (default: 735).
  -m <unsigned integer>:  The number of milliseconds per chunk (default: unused).
  -c <unsigned integer>:  The number of channels of the audio data (default: 2).
  -i <string>:            The name of the input audio file (default: none).
  -o <string>:            The name of the output audio file (default: none).
  -b <unsigned integer>:  The number of buffers used for audio processing (default: 2).
  -d <unsigned integer>:  The processing delay in microseconds (default: 0).
  -j <unsigned integer>:  The jitter in microseconds (default: 0).
  -x:                     Disables simulation and processes audio data as fast as possible.
  -h:                     Displays this help message and terminates.
```

The framework can operate in 4 modes:
  * direct from microphone to speaker processing (neither input file nor output file is specified),
  * from microphone to file processing (only output file is specified),
  * from file to speaker processing (only input file is specified),
  * from file to file simulation of real time processing (both input and output files are specified).

## Experiments
The `processed_files` directory contains results of simulations with different settings. The files are named with the following pattern `b<number of buffers>_d<processing delay in microseconds>_j<jitter in microseconds>.wav` capturing the used settings. The used window size was always `33.333 ms` (`33333 us`). The output thread writes zeros to the output file, when data are not available (missed deadline), causing hearable flickering, e.g. `processed_files/b4_d33700_j0.wav` or `processed_files/b2_d33600_j3000.wav` are great examples of the phenomenon. We can hear that the flickering appears in waves, which is caused by that the audio processing catches up with the output, when enough zeros are written to the output file.
