// mpic++ simple_echo.cpp -O3 -std=c++17 /usr/local/lib/libportaudio.a -lrt -lm -lasound -pthread -o simple_echo

#include "portaudio.h"
#include "mpi.h"

#include <iostream>

#define SAMPLE_RATE 44100
#define NUM_CHANNELS 2
#define NUM_SECONDS 4
#define NUM_SAMPLES 512

int main()
{
    MPI_Init(nullptr, nullptr);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    float inputBuffer[NUM_SAMPLES * 2 + 100] = {0, };
    float outputBuffer[NUM_SAMPLES * 2 + 100] = {0, };

    if (rank == 0)
    {
        Pa_Initialize();
        PaError err = paNoError;

        PaDeviceIndex inputDevice = Pa_GetDefaultInputDevice();
        PaDeviceIndex outputDevice = Pa_GetDefaultOutputDevice();

        PaStreamParameters inputParameters;
        inputParameters.device = inputDevice;
        inputParameters.channelCount = NUM_CHANNELS;
        inputParameters.sampleFormat = paFloat32;
        inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
        inputParameters.hostApiSpecificStreamInfo = nullptr;

        PaStreamParameters outputParameters;
        outputParameters.device = outputDevice;
        outputParameters.channelCount = NUM_CHANNELS;
        outputParameters.sampleFormat = paFloat32;
        outputParameters.suggestedLatency = Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;
        outputParameters.hostApiSpecificStreamInfo = nullptr;

        PaStream *stream;
        err = Pa_OpenStream(&stream, &inputParameters, &outputParameters, SAMPLE_RATE, NUM_SAMPLES, paClipOff, nullptr, nullptr);

        if (err != paNoError)
        {
            std::cerr << "Error: " << Pa_GetErrorText(err) << std::endl;
            return 1;
        }

        err = Pa_StartStream(stream);
        if (err != paNoError)
        {
            std::cerr << "Error: " << Pa_GetErrorText(err) << std::endl;
            return 1;
        }

        for (int i = 0; i < 4; i++)
        {
            Pa_WriteStream(stream, outputBuffer, NUM_SAMPLES);
        }

        for (int i = 0; i < 5000; i++)
        {
            Pa_ReadStream(stream, inputBuffer, NUM_SAMPLES);
            MPI_Send(inputBuffer, NUM_SAMPLES * 2, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);

            MPI_Recv(outputBuffer, NUM_SAMPLES * 2, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            Pa_WriteStream(stream, outputBuffer, NUM_SAMPLES);
        }

        Pa_StopStream(stream);
        Pa_Terminate();
    }
    else if (rank == 1)
    {
        float lastRight = 0;
        float lastLeft = 0;
        float beta = 0.04;
        for (int i = 0; i < 5000; i++)
        {
            MPI_Recv(inputBuffer, NUM_SAMPLES * 2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int j = 0; j < NUM_SAMPLES * 2; j += 2)
            {
            
            #if 1
                lastRight = lastRight - (beta * (lastRight - inputBuffer[j]));
                lastLeft = lastLeft - (beta * (lastLeft - inputBuffer[j + 1]));
                outputBuffer[j] = lastRight;
                outputBuffer[j + 1] = lastLeft;
            #else
                outputBuffer[j] = inputBuffer[j];
                outputBuffer[j + 1] = inputBuffer[j + 1];
            #endif
            }

            MPI_Send(outputBuffer, NUM_SAMPLES * 2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}