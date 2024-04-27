#include "client.h"

using namespace std;

Client::Client(int argc, char **argv) : ClientServer(argc, argv)
{
    PaError err = Pa_Initialize();
    if (err != paNoError)
    {
        cerr << "PortAudio initialization error: " << Pa_GetErrorText(err) << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    PaDeviceIndex inputDevice = Pa_GetDefaultInputDevice();
    PaDeviceIndex outputDevice = Pa_GetDefaultOutputDevice();

    PaStreamParameters inputParameters;
    inputParameters.device = inputDevice;
    inputParameters.channelCount = _channels;
    inputParameters.sampleFormat = _samplingDatatype;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = nullptr;

    PaStreamParameters outputParameters;
    outputParameters.device = outputDevice;
    outputParameters.channelCount = _channels;
    outputParameters.sampleFormat = _samplingDatatype;
    outputParameters.suggestedLatency = Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;
    outputParameters.hostApiSpecificStreamInfo = nullptr;

    err = Pa_OpenStream(&_stream, &inputParameters, &outputParameters, _samplingRate, _samplesPerChunk, paClipOff, nullptr, nullptr);
    if (err != paNoError)
    {
        cerr << "PortAudio opening stream error: " << Pa_GetErrorText(err) << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    switch (_samplingDatatype)
    {
        case paFloat32:
            _inputBuffer.f32 = new float[_samplesPerChunk * _channels];
            _outputBuffer.f32 = new float[_samplesPerChunk * _channels];
            break;

        case paInt32:
            _inputBuffer.i32 = new int32_t[_samplesPerChunk * _channels];
            _outputBuffer.i32 = new int32_t[_samplesPerChunk * _channels];
            break;

        case paInt16:
            _inputBuffer.i16 = new int16_t[_samplesPerChunk * _channels];
            _outputBuffer.i16 = new int16_t[_samplesPerChunk * _channels];
            break;

        case paInt8:
            _inputBuffer.i8 = new int8_t[_samplesPerChunk * _channels];
            _outputBuffer.i8 = new int8_t[_samplesPerChunk * _channels];
            break;

        case paUInt8:
            _inputBuffer.u8 = new uint8_t[_samplesPerChunk * _channels];
            _outputBuffer.u8 = new uint8_t[_samplesPerChunk * _channels];
            break;
    }

    _bufferByteSize = _samplesPerChunk * _channels * Pa_GetSampleSize(_samplingDatatype);
}

Client::~Client()
{
    Pa_CloseStream(_stream);
    Pa_Terminate();

    switch (_samplingDatatype)
    {
        case paFloat32:
            delete[] _inputBuffer.f32;
            delete[] _outputBuffer.f32;
            break;

        case paInt32:
            delete[] _inputBuffer.i32;
            delete[] _outputBuffer.i32;
            break;

        case paInt16:
            delete[] _inputBuffer.i16;
            delete[] _outputBuffer.i16;
            break;

        case paInt8:
            delete[] _inputBuffer.i8;
            delete[] _outputBuffer.i8;
            break;

        case paUInt8:
            delete[] _inputBuffer.u8;
            delete[] _outputBuffer.u8;
            break;
    }
}

void Client::parseArguments(int argc, char **argv)
{
    ClientServer::parseArguments(argc, argv);

    // convert arguments to a vector of strings
    vector<string> arguments(argv, argv + argc);
    size_t idx = 0;

    // lambda function for parsing an integer argument
    auto checkNextArgument = [&]()
    {
        idx++;
        if (idx == arguments.size())
        {
            cerr << "No argument provided for the '" << arguments[idx - 1] << "' switch." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    };
    

    for (; idx < arguments.size(); idx)
    {
        if (arguments[idx] == "-d")
        {
            checkNextArgument();
            if (arguments[idx] == "f32")
            {
                _samplingDatatype = paFloat32;
            }
            else if (arguments[idx] == "i32")
            {
                _samplingDatatype = paInt32;
            }
            else if (arguments[idx] == "i16")
            {
                _samplingDatatype = paInt16;
            }
            else if (arguments[idx] == "i8")
            {
                _samplingDatatype = paInt8;
            }
            else if (arguments[idx] == "u8")
            {
                _samplingDatatype = paUInt8;
            }
            else
            {
                if (_rank == ROOT)
                {
                    cerr << "Invalid argument for the '-d' switch." << endl;
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
}

void Client::startSendChunk()
{
    MPI_Isend(_inputBuffer.any, _bufferByteSize, MPI_BYTE, SERVER_RANK, 0, _clientServerComm, &_sendRequest);
}

void Client::startReceiveChunk()
{
    MPI_Irecv(_outputBuffer.any, _bufferByteSize, MPI_BYTE, SERVER_RANK, 0, _clientServerComm, &_receiveRequest);
}

void Client::awaitSendChunk()
{
    MPI_Wait(&_sendRequest, MPI_STATUS_IGNORE);
}

void Client::awaitReceiveChunk()
{
    MPI_Wait(&_receiveRequest, MPI_STATUS_IGNORE);
}

void Client::run()
{
    PaError err = Pa_StartStream(_stream);
    if (err != paNoError)
    {
        cerr << "PortAudio starting stream error: " << Pa_GetErrorText(err) << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < INITIAL_WRITE_PADDING; i++)
    {
        Pa_WriteStream(_stream, _outputBuffer.f32, _samplesPerChunk);
    }

    cout << "Press any key to stop the audio processing..." << endl;
    while (!getch())
    {
        startReceiveChunk();

        Pa_ReadStream(_stream, _inputBuffer.f32, _samplesPerChunk);
        startSendChunk();

        awaitReceiveChunk();
        Pa_WriteStream(_stream, _outputBuffer.f32, _samplesPerChunk);

        awaitSendChunk();
    }
}
