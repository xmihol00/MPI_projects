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
}

void ClientServer::parseArguments(int argc, char **argv)
{
    // convert arguments to a vector of strings
    vector<string> arguments(argv, argv + argc);
    size_t idx = 0;

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
    

    for (; idx < arguments.size(); idx)
    {
        if (arguments[idx] == "-s")
        {
            checkNextArgument();
            _samplingRate = parseInt();
        }
        else if (arguments[idx] == "-n")
        {
            _samplesPerChunk = parseInt();
        }
        else if (arguments[idx] == "-m")
        {
            _millisecondsPerChunk = parseInt();
        }
        else if (arguments[idx] == "-c")
        {
            _channels = parseInt();
        }
    }

    if (_millisecondsPerChunk)
    {
        _samplesPerChunk = _samplingRate * _millisecondsPerChunk / 1000;
    }
}
