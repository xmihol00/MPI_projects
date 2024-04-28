#include "client.h"

using namespace std;

Client::Client(int argc, char **argv) : ClientServer(argc, argv)
{
    parseArguments(argc, argv);

    // allocate memory for the buffers
    _currentInputBuffer = new float[_samplesPerChunk * _channels]();
    _currentOutputBuffer = new float[_samplesPerChunk * _channels]();
    _nextInputBuffer = new float[_samplesPerChunk * _channels]();
    _nextOutputBuffer = new float[_samplesPerChunk * _channels]();
    _inputBuffer = new float[_samplesPerChunk * _channels]();
    for (uint32_t i = 0; i < _numberOfUsedBuffers; i++)
    {
        _outputBuffers.push_back(new float[_samplesPerChunk * _channels]());
    }
    
    _bufferByteSize = _samplesPerChunk * _channels * sizeof(float);

    if (_inputFileName.empty() || _outputFileName.empty()) // audio library will be used, processing will be done using the microphone and/or the speakers
    {
        PaError err = Pa_Initialize();
        if (err != paNoError)
        {
            cerr << "PortAudio initialization error: " << Pa_GetErrorText(err) << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        PaStreamParameters *inputParametersPtr = nullptr;
        PaStreamParameters *outputParametersPtr = nullptr;
        PaStreamParameters inputParameters;
        PaStreamParameters outputParameters;

        if (_inputFileName.empty()) // input from the microphone
        {
            PaDeviceIndex inputDevice = Pa_GetDefaultInputDevice();

            inputParameters.device = inputDevice;
            inputParameters.channelCount = _channels;
            inputParameters.sampleFormat = paFloat32;
            inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
            inputParameters.hostApiSpecificStreamInfo = nullptr;

            inputParametersPtr = &inputParameters;
        }
        

        if (_outputFileName.empty()) // output to the speakers
        {
            PaDeviceIndex outputDevice = Pa_GetDefaultOutputDevice();

            outputParameters.device = outputDevice;
            outputParameters.channelCount = _channels;
            outputParameters.sampleFormat = paFloat32;
            outputParameters.suggestedLatency = Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;
            outputParameters.hostApiSpecificStreamInfo = nullptr;

            outputParametersPtr = &outputParameters;
        }
        
        err = Pa_OpenStream(&_stream, inputParametersPtr, outputParametersPtr, _samplingRate, _samplesPerChunk, paClipOff, nullptr, nullptr);
        if (err != paNoError)
        {
            cerr << "PortAudio opening stream error: " << Pa_GetErrorText(err) << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (!_inputFileName.empty()) // input from a file
    {
        _inputFile = SndfileHandle(_inputFileName, SFM_READ, SF_FORMAT_WAV | SF_FORMAT_PCM_16, _channels, _samplingRate);
        if (_inputFile.error())
        {
            cerr << "Error opening input file: " << _inputFileName << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (!_outputFileName.empty()) // output to a file
    {
        _outputFile = SndfileHandle(_outputFileName, SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_16, _channels, _samplingRate);
        if (_outputFile.error())
        {
            cerr << "Error opening output file: " << _outputFileName << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

Client::~Client()
{
    if (_inputFileName.empty() || _outputFileName.empty()) // audio library was used
    {
        Pa_CloseStream(_stream);
        Pa_Terminate();
    }

    // deallocate memory for the buffers
    delete[] _nextOutputBuffer;
    delete[] _nextInputBuffer;
    delete[] _currentOutputBuffer;
    delete[] _currentInputBuffer;
    delete[] _inputBuffer;
    for (uint32_t i = 0; i < _numberOfUsedBuffers; i++)
    {
        delete[] _outputBuffers[i];
    }
    
    ClientServer::~ClientServer();
}

void Client::parseArguments(int argc, char **argv)
{
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

    // lambda function for parsing an integer argument
    auto parseInt = [&]() -> int
    {
        int value = 0;
        try
        {
            value = stoi(arguments[idx]);
        }
        catch (const invalid_argument& e)
        {
            cerr << "Invalid argument for the '" << arguments[idx - 1] << "' switch." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        catch (const out_of_range& e)
        {
            cerr << "Argument for the '" << arguments[idx - 1] << "' switch is out of range." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (value < 0)
        {
            cerr << "Argument for the '" << arguments[idx - 1] << "' switch must be a non-negative number." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        return value;
    };

    for (; idx < arguments.size(); idx++)
    {
        if (arguments[idx] == "-i")      // input file name
        {
            checkNextArgument();
            _inputFileName = arguments[idx];
        }
        else if (arguments[idx] == "-o") // output file name
        {
            checkNextArgument();
            _outputFileName = arguments[idx];
        }
        else if (arguments[idx] == "-b") // number of used buffers
        {
            checkNextArgument();
            _numberOfUsedBuffers = parseInt();
            if (_numberOfUsedBuffers < 2)
            {
                cerr << "Number of used buffers must be at least 2." << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            _numberOfUsedBuffers++;
        }
        else if (arguments[idx] == "-x") // disable simulation
        {
            _noSimulation = true;
        }
    }
}

bool Client::enterNotPressed()
{   
    fd_set fdSet;
    FD_ZERO(&fdSet);
    FD_SET(STDIN_FILENO, &fdSet);
    timeval timeout{tv_sec: 0, tv_usec: 0};
    return select(1, &fdSet, nullptr, nullptr, &timeout) == 0; // check if the Enter key was pressed
}

void Client::sendChunk()
{
    MPI_Send(_inputBuffer, _bufferByteSize, MPI_BYTE, SERVER_RANK, VALID_TAG, _clientServerComm);
}

void Client::receiveChunk()
{
    MPI_Recv(_outputBuffers[_outputBufferWriteIdx], _bufferByteSize, MPI_BYTE, SERVER_RANK, VALID_TAG, _clientServerComm, MPI_STATUS_IGNORE);
}

void Client::startSendChunk()
{
    MPI_Isend(_currentInputBuffer, _bufferByteSize, MPI_BYTE, SERVER_RANK, VALID_TAG, _clientServerComm, &_sendRequest);
}

void Client::startReceiveChunk()
{
    MPI_Irecv(_nextOutputBuffer, _bufferByteSize, MPI_BYTE, SERVER_RANK, VALID_TAG, _clientServerComm, &_receiveRequest);
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

void Client::stopProcessing()
{
    MPI_Send(_currentInputBuffer, 1, MPI_BYTE, SERVER_RANK, TERMINATING_TAG, _clientServerComm);
}

void Client::run()
{
    if (_inputFileName.empty() || _outputFileName.empty()) // processing will be done using the microphone and/or the speakers
    {
        PaError err = Pa_StartStream(_stream); // start streaming the audio from/to the HW
        if (err != paNoError)
        {
            cerr << "PortAudio starting stream error: " << Pa_GetErrorText(err) << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (_inputFileName.empty() && _outputFileName.empty()) // processing from the microphone to the speakers
    {
        for (uint32_t i = 0; i < _numberOfUsedBuffers; i++) // fill in the audio output with initial silence
        {
            Pa_WriteStream(_stream, _currentOutputBuffer, _samplesPerChunk);
        }

        cout << "Press Enter to stop the audio processing..." << endl;
        
        sendChunk(); // initiate the communication with the server, the client must be ahead to ensure overlaying of communication with processing
        startSendChunk();
        startReceiveChunk();
        do
        {
            Pa_ReadStream(_stream, _nextInputBuffer, _samplesPerChunk); // read audio from the microphone
            awaitSendChunk(); // wait for the previous chunk to be sent
            swap(_currentInputBuffer, _nextInputBuffer);   // ping-pong buffering
            startSendChunk(); // start sending the current chunk

            awaitReceiveChunk(); // wait for the previous chunk to be received
            swap(_currentOutputBuffer, _nextOutputBuffer); // ping-pong buffering
            startReceiveChunk(); // start receiving the next chunk
            Pa_WriteStream(_stream, _currentOutputBuffer, _samplesPerChunk); // write audio to the speakers
        }
        while (enterNotPressed());
        // ensure the server catches up
        awaitSendChunk();
        startReceiveChunk();
        awaitReceiveChunk();
        stopProcessing(); // send the terminating message to the server
        cout << "Audio processing stopped." << endl;
    }
    else if (_outputFileName.empty())
    {
        sf_count_t samplesRead = 0;
        for (uint32_t i = 0; i < _numberOfUsedBuffers - 1; i++)
        {
            Pa_WriteStream(_stream, _currentOutputBuffer, _samplesPerChunk);
        }

        cout << "Replaying audio file: " << _inputFileName << " press Enter to stop." << endl;
        startSendChunk();
        awaitSendChunk();
        startSendChunk();
        startReceiveChunk();
        do
        {
            samplesRead = _inputFile.readf(_currentInputBuffer, _samplesPerChunk); // read audio from the file
            awaitSendChunk();
            swap(_currentInputBuffer, _nextInputBuffer);
            startSendChunk();

            awaitReceiveChunk();
            swap(_currentOutputBuffer, _nextOutputBuffer);
            startReceiveChunk();    
            Pa_WriteStream(_stream, _currentOutputBuffer, _samplesPerChunk); // write audio to the speakers
        }
        while (samplesRead && enterNotPressed());
        awaitSendChunk();
        startReceiveChunk();
        awaitReceiveChunk();
        stopProcessing();
        cout << "Audio file replayed." << endl;
    }
    else if (_inputFileName.empty())
    {
        for (uint32_t i = 0; i < _numberOfUsedBuffers - 1; i++)
        {
            Pa_WriteStream(_stream, _currentOutputBuffer, _samplesPerChunk);
        }

        cout << "Recording audio to file: " << _outputFileName << " press Enter to stop." << endl;
        startSendChunk();
        startReceiveChunk();
        do
        {
            Pa_ReadStream(_stream, _nextInputBuffer, _samplesPerChunk); // read audio from the microphone
            awaitSendChunk();
            swap(_currentInputBuffer, _nextInputBuffer);
            startSendChunk();

            awaitReceiveChunk();
            swap(_currentOutputBuffer, _nextOutputBuffer);
            startReceiveChunk();
            _outputFile.writef(_currentOutputBuffer, _samplesPerChunk); // write audio to the file
        }
        while (enterNotPressed());
        awaitSendChunk();
        startReceiveChunk();
        awaitReceiveChunk();
        stopProcessing();
        cout << "Audio recorded." << endl;
    }
    else if (_noSimulation)
    {
        cout << "Converting audio file: " << _inputFileName << " to: " << _outputFileName << endl;
        sendChunk();
        while (_inputFile.readf(_inputBuffer, _samplesPerChunk)) // read audio from the file
        {
            sendChunk();
            receiveChunk();
            _outputFile.writef(_outputBuffers[_outputBufferWriteIdx], _samplesPerChunk); // immediately write the processed audio to the output file
        }
        receiveChunk();
        stopProcessing();
        cout << "Audio file converted." << endl;
    }
    else
    {
        // shared variables between the threads below to simulate real-time audio processing
        bool run = true;
        size_t fileSize = _inputFile.frames();
        int64_t bufferWrites = 0;
        int64_t bufferReads = 0;

        sendChunk(); // ensure the server is ahead
        // fill in the output buffers first
        for (uint32_t i = 0; i < _numberOfUsedBuffers - 1; i++)
        {
            _inputFile.readf(_inputBuffer, _samplesPerChunk);
            sendChunk();
            receiveChunk();
            _outputBufferWriteIdx++;
            bufferWrites++;
        }

        cout << "Simulating real-time audio processing, press Enter to stop." << endl;
        sf_count_t samplesRead = 0;
        omp_set_num_threads(2); // use 2 threads for the simulation, one for reading the audio file and one for writing to the output file
        #pragma omp parallel sections
        {
            #pragma omp section // reading thread
            {
                do
                {
                    bufferWrites++;
                    while (bufferWrites - bufferReads >= _numberOfUsedBuffers - 1) // wait for the output buffers to be read
                    {
                        this_thread::sleep_for(chrono::microseconds(10));
                    }

                    samplesRead = _inputFile.readf(_inputBuffer, _samplesPerChunk); // read audio from the file
                    sendChunk();
                    _outputBufferWriteIdx++;
                    _outputBufferWriteIdx %= _numberOfUsedBuffers;
                    receiveChunk();
                }
                while (samplesRead && enterNotPressed());
                run = !samplesRead;
                _outputBufferWriteIdx++;
                _outputBufferWriteIdx %= _numberOfUsedBuffers;
                receiveChunk();
                stopProcessing();
                cout << "Reading audio file stopped." << endl;
            }

            #pragma omp section // writing thread
            {
                size_t writtenFrames = 0;
                double secondsPerFrame = 1.0 / _samplingRate;
                auto startTime = chrono::high_resolution_clock::now(); // start time of the simulation by the writing thread

                while (run && writtenFrames < fileSize)
                {
                    for (uint32_t i = 0; i < _samplesPerChunk; i++)
                    {
                        _outputFile.writef(_outputBuffers[_outputBufferReadIdx] + i * _channels, 1);
                        writtenFrames++;
                        
                        // simulate real-time playback, i.e. writing to the file at the same rate as the audio is played
                        auto currentTime = chrono::high_resolution_clock::now();
                        auto elapsedTime = chrono::duration_cast<chrono::duration<double>>(currentTime - startTime).count();
                        if (elapsedTime < secondsPerFrame * writtenFrames)
                        {
                            // simulate real-time playback by waiting for the appropriate amount of time
                            this_thread::sleep_for(chrono::duration<double>(secondsPerFrame * writtenFrames - elapsedTime));
                        }
                    }
                    
                    memset(_outputBuffers[_outputBufferReadIdx], 0.0, _bufferByteSize); // ensure that if we run out of data, old samples are not replayed
                    _outputBufferReadIdx++;
                    _outputBufferReadIdx %= _numberOfUsedBuffers;
                    bufferReads++;
                }
                cout << "Writing audio file stopped." << endl;
            }
        }
        cout << "Simulation stopped." << endl;
    }
}
