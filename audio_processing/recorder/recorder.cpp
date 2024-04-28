// g++ recorder.cpp /usr/local/lib/libportaudio.a -lsndfile -lrt -lm -lasound -pthread -O3 -std=c++20 -o record

#include <portaudio.h>
#include <sndfile.hh>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

using namespace std;

#define SAMPLING_RATE 22050
#define NUM_CHANNELS 2
#define NUM_SAMPLES 512

bool enterNotPressed()
{
    fd_set fdSet;
    FD_ZERO(&fdSet);
    FD_SET(STDIN_FILENO, &fdSet);
    timeval timeout{tv_sec: 0, tv_usec: 0};
    return select(1, &fdSet, nullptr, nullptr, &timeout) == 0;
}

int main(int argc, char **argv)
{
    float buffer[NUM_SAMPLES * NUM_CHANNELS] = {0, };

    SndfileHandle file("recording.wav", SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_16, NUM_CHANNELS, SAMPLING_RATE);

    Pa_Initialize();
    PaError err = paNoError;

    PaDeviceIndex inputDevice = Pa_GetDefaultInputDevice();

    PaStreamParameters inputParameters;
    inputParameters.device = inputDevice;
    inputParameters.channelCount = NUM_CHANNELS;
    inputParameters.sampleFormat = paFloat3232;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = nullptr;

    PaStream *stream;
    err = Pa_OpenStream(&stream, &inputParameters, nullptr, SAMPLING_RATE, NUM_SAMPLES, paClipOff, nullptr, nullptr);

    if (err != paNoError)
    {
        cerr << "Error: " << Pa_GetErrorText(err) << endl;
        return 1;
    }

    err = Pa_StartStream(stream);
    if (err != paNoError)
    {
        cerr << "Error: " << Pa_GetErrorText(err) << endl;
        return 1;
    }

    cout << "Waiting for 1s to stabilize..." << endl;
    for (int i = 0; i < SAMPLING_RATE; i += NUM_SAMPLES)
    {
        Pa_ReadStream(stream, buffer, NUM_SAMPLES);
    }

    cout << "Press Enter to stop recording." << endl;
    while (enterNotPressed())
    {
        Pa_ReadStream(stream, buffer, NUM_SAMPLES);
        file.write(buffer, NUM_SAMPLES * NUM_CHANNELS);
    }
    cout << "Recording stopped." << endl;

    Pa_StopStream(stream);
    Pa_Terminate();

    return 0;
}