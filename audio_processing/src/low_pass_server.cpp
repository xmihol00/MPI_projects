#include "low_pass_server.h"

using namespace std;

LowPassServer::LowPassServer(int argc, char **argv) : Server(argc, argv)
{
    if (_channels != EXPECTED_CHANNELS)
    {
        cerr << "Low pass server expects " << EXPECTED_CHANNELS << " channels, but received " << _channels << " channels." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    _currentInputBuffer = new float[_samplesPerChunk * _channels]();
    _currentOutputBuffer = new float[_samplesPerChunk * _channels]();
    _nextInputBuffer = new float[_samplesPerChunk * _channels]();
    _nextOutputBuffer = new float[_samplesPerChunk * _channels]();
    _bufferByteSize = _samplesPerChunk * _channels * sizeof(float);
}

LowPassServer::~LowPassServer()
{
    delete[] _nextOutputBuffer;
    delete[] _nextInputBuffer;
    delete[] _currentOutputBuffer;
    delete[] _currentInputBuffer;

    Server::~Server();
}

void LowPassServer::parseArguments(int argc, char **argv)
{
    // convert arguments to a vector of strings
    vector<string> arguments(argv, argv + argc);
    size_t idx = 1;

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
        if (arguments[idx] == "-d")      // processing delay
        {
            checkNextArgument();
            _processingDelay = parseInt();
        }
        else if (arguments[idx] == "-j") // jitter
        {
            checkNextArgument();
            _jitter = parseInt();
            _jitter++;  // ensure jitter is never zero to avoid division by zero
            _jitterShift = _jitter / 2;
        }
        else if (arguments[idx] == "-h") // help
        {
            MPI_Finalize();
            exit(0);
        }
    }
}

void LowPassServer::processChunk()
{
    for (uint32_t i = 0; i < _samplesPerChunk * _channels; i += _channels)
    {
        _lastRightSample = _lastRightSample - (LOW_PASS_COEFFICIENT * (_lastRightSample - _currentInputBuffer[i]));
        _lastLeftSample = _lastLeftSample - (LOW_PASS_COEFFICIENT * (_lastLeftSample - _currentInputBuffer[i + 1]));

        _currentOutputBuffer[i] = _lastRightSample * 1.25f;
        _currentOutputBuffer[i + 1] = _lastLeftSample * 1.25f;
    }

    uint32_t delay = _processingDelay + (rand() % _jitter) - _jitterShift;
    usleep(delay);
}
