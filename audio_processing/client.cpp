#include "client.h"

using namespace std;

Client::Client(int argc, char **argv) : ClientServer(argc, argv)
{
    switch (_samplingDatatype)
    {
        case paFloat32:
            _currentInputBuffer.f32 = new float[_samplesPerChunk * _channels]();
            _currentOutputBuffer.f32 = new float[_samplesPerChunk * _channels]();
            _nextInputBuffer.f32 = new float[_samplesPerChunk * _channels]();
            _nextOutputBuffer.f32 = new float[_samplesPerChunk * _channels]();
            break;

        case paInt32:
            _currentInputBuffer.i32 = new int32_t[_samplesPerChunk * _channels]();
            _currentOutputBuffer.i32 = new int32_t[_samplesPerChunk * _channels]();
            _nextInputBuffer.i32 = new int32_t[_samplesPerChunk * _channels]();
            _nextOutputBuffer.i32 = new int32_t[_samplesPerChunk * _channels]();
            break;

        case paInt16:
            _currentInputBuffer.i16 = new int16_t[_samplesPerChunk * _channels]();
            _currentOutputBuffer.i16 = new int16_t[_samplesPerChunk * _channels]();
            _nextInputBuffer.i16 = new int16_t[_samplesPerChunk * _channels]();
            _nextOutputBuffer.i16 = new int16_t[_samplesPerChunk * _channels]();
            break;

        case paInt8:
            _currentInputBuffer.i8 = new int8_t[_samplesPerChunk * _channels]();
            _currentOutputBuffer.i8 = new int8_t[_samplesPerChunk * _channels]();
            _nextInputBuffer.i8 = new int8_t[_samplesPerChunk * _channels]();
            _nextOutputBuffer.i8 = new int8_t[_samplesPerChunk * _channels]();
            break;

        case paUInt8:
            _currentInputBuffer.u8 = new uint8_t[_samplesPerChunk * _channels]();
            _currentOutputBuffer.u8 = new uint8_t[_samplesPerChunk * _channels]();
            _nextInputBuffer.u8 = new uint8_t[_samplesPerChunk * _channels]();
            _nextOutputBuffer.u8 = new uint8_t[_samplesPerChunk * _channels]();
            break;
    }

    _bufferByteSize = _samplesPerChunk * _channels * Pa_GetSampleSize(_samplingDatatype);

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
}

Client::~Client()
{
    Pa_CloseStream(_stream);
    Pa_Terminate();

    switch (_samplingDatatype)
    {
        case paFloat32:
            delete[] _nextOutputBuffer.f32;
            delete[] _nextInputBuffer.f32;
            delete[] _currentOutputBuffer.f32;
            delete[] _currentInputBuffer.f32;
            break;

        case paInt32:
            delete[] _nextOutputBuffer.i32;
            delete[] _nextInputBuffer.i32;
            delete[] _currentOutputBuffer.i32;
            delete[] _currentInputBuffer.i32;
            break;

        case paInt16:
            delete[] _nextOutputBuffer.i16;
            delete[] _nextInputBuffer.i16;
            delete[] _currentOutputBuffer.i16;
            delete[] _currentInputBuffer.i16;
            break;

        case paInt8:
            delete[] _nextOutputBuffer.i8;
            delete[] _nextInputBuffer.i8;
            delete[] _currentOutputBuffer.i8;
            delete[] _currentInputBuffer.i8;
            break;

        case paUInt8:
            delete[] _nextOutputBuffer.u8;
            delete[] _nextInputBuffer.u8;
            delete[] _currentOutputBuffer.u8;
            delete[] _currentInputBuffer.u8;
            break;
    }

    ClientServer::~ClientServer();
}

void Client::parseArguments(int argc, char **argv)
{
    ClientServer::parseArguments(argc, argv);

    // convert arguments to a vector of strings
    vector<string> arguments(argv, argv + argc);
    size_t idx = 1;

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
    

    for (; idx < arguments.size(); idx++)
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

bool Client::enterNotPressed()
{   
    fd_set fdSet;
    FD_ZERO(&fdSet);
    FD_SET(STDIN_FILENO, &fdSet);
    return select(1, &fdSet, nullptr, nullptr, &_terminalTimeout) == 0;
}

void Client::startSendChunk()
{
    MPI_Isend(_currentInputBuffer.any, _bufferByteSize, MPI_BYTE, SERVER_RANK, VALID_TAG, _clientServerComm, &_sendRequest);
}

void Client::startReceiveChunk()
{
    MPI_Irecv(_nextOutputBuffer.any, _bufferByteSize, MPI_BYTE, SERVER_RANK, VALID_TAG, _clientServerComm, &_receiveRequest);
}

bool Client::awaitSendChunk()
{
    MPI_Wait(&_sendRequest, MPI_STATUS_IGNORE); // FIXME, should check for errors
    return true;
}

bool Client::awaitReceiveChunk()
{
    MPI_Wait(&_receiveRequest, MPI_STATUS_IGNORE); // FIXME, should check for errors
    return true;
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
        Pa_WriteStream(_stream, _currentOutputBuffer.any, _samplesPerChunk);
    }

    cout << "Press enter to stop the audio processing..." << endl;
    startSendChunk();
    while (enterNotPressed())
    {
        startReceiveChunk();

        awaitSendChunk();
        swap(_currentOutputBuffer.any, _nextOutputBuffer.any);
        Pa_ReadStream(_stream, _currentInputBuffer.any, _samplesPerChunk);
        startSendChunk();

        awaitReceiveChunk();
        swap(_currentInputBuffer.any, _nextInputBuffer.any);
        Pa_WriteStream(_stream, _currentOutputBuffer.any, _samplesPerChunk);
    }
    startReceiveChunk();
    MPI_Send(_currentInputBuffer.any, 1, MPI_BYTE, SERVER_RANK, TERMINATING_TAG, _clientServerComm);

    cout << "Audio processing stopped." << endl;
}
