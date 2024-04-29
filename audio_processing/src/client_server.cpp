#include "client_server.h"

using namespace std;

ClientServer::ClientServer(int argc, char **argv)
{
    MPI_Comm_dup(MPI_COMM_WORLD, &_clientServerComm);
    MPI_Comm_rank(_clientServerComm, &_rank);

    parseArguments(argc, argv);
}

ClientServer::~ClientServer()
{
    if (_clientServerComm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&_clientServerComm);
    }

    if (_sendRequest != MPI_REQUEST_NULL)
    {
        MPI_Status status;
        int flag;
        MPI_Request_get_status(_sendRequest, &flag, &status);
        if (!flag)
        {
            MPI_Cancel(&_sendRequest);
        }
    }

    if (_receiveRequest != MPI_REQUEST_NULL)
    {
        MPI_Status status;
        int flag;
        MPI_Request_get_status(_receiveRequest, &flag, &status);
        if (!flag)
        {
            MPI_Cancel(&_receiveRequest);
        }
    }
}

void ClientServer::parseArguments(int argc, char **argv)
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
            if (_rank == ROOT)
            {
                cerr << "Invalid argument for the '" << arguments[idx - 1] << "' switch." << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        catch (const out_of_range& e)
        {
            if (_rank == ROOT)
            {
                cerr << "Argument for the '" << arguments[idx - 1] << "' switch is out of range." << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (value < 0)
        {
            if (_rank == ROOT)
            {
                cerr << "Argument for the '" << arguments[idx - 1] << "' switch must be a non-negative number." << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        return value;
    };

    auto checkNextArgument = [&]()
    {
        idx++;
        if (idx == arguments.size())
        {
            if (_rank == ROOT)
            {
                cerr << "No argument provided for the '" << arguments[idx - 1] << "' switch." << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    };
    
    for (; idx < arguments.size(); idx++)
    {
        if (arguments[idx] == "-s")      // sampling rate
        {
            checkNextArgument();
            _samplingRate = parseInt();
        }
        else if (arguments[idx] == "-n") // samples per chunk
        {
            checkNextArgument();
            _samplesPerChunk = parseInt();
        }
        else if (arguments[idx] == "-m") // milliseconds per chunk
        {
            checkNextArgument();
            _millisecondsPerChunk = parseInt();
        }
        else if (arguments[idx] == "-c") // number of channels
        {
            checkNextArgument();
            _channels = parseInt();
        }
        else if (arguments[idx] == "-h") // help message
        {
            if (_rank == ROOT)
            {
                cout << "Usage: mpiexec -np 2 ./rtap [-s <sampling rate>] [-n <samples per chunk>] [-m <milliseconds per chunk>]" << endl;
                cout << "                            [-c <channels>] [-i <input file>] [-o <output file>] [-b <number of buffers>]" << endl;
                cout << "                            [-d <processing delay>] [-j <jitter>] [-x] [-h]" << endl;
                cout << "Options:" << endl;
                cout << "  -s <unsigned integer>:  The sampling rate of the audio data in Hz (default: 22050)." << endl;
                cout << "  -n <unsigned integer>:  The number of samples per chunk (window size) (default: 735)." << endl;
                cout << "  -m <unsigned integer>:  The number of milliseconds per chunk (default: unused)." << endl;
                cout << "  -c <unsigned integer>:  The number of channels of the audio data (default: 2)." << endl;
                cout << "  -i <string>:            The name of the input audio file (default: none)." << endl;
                cout << "  -o <string>:            The name of the output audio file (default: none)." << endl;
                cout << "  -b <unsigned integer>:  The number of buffers used for audio processing (default: 2)." << endl;
                cout << "  -d <unsigned integer>:  The processing delay in microseconds (default: 0)." << endl;
                cout << "  -j <unsigned integer>:  The jitter in microseconds (default: 0)." << endl;
                cout << "  -x:                     Disables simulation and processes audio data as fast as possible." << endl;
                cout << "  -h:                     Displays this help message and terminates." << endl;
            }

            MPI_Finalize();
            exit(0);
        }
    }

    if (_millisecondsPerChunk)
    {
        _samplesPerChunk = _samplingRate * _millisecondsPerChunk / 1000;
    }
}
