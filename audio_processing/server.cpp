#include "server.h"

using namespace std;

Server::Server(int argc, char **argv) : ClientServer(argc, argv) 
{ 
    parseArguments(argc, argv);
}

Server::~Server()
{
    ClientServer::~ClientServer();
}

void Server::parseArguments(int argc, char **argv)
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
        if (arguments[idx] == "-d")
        {
            checkNextArgument();
            _processingDelay = parseInt();
            _processingDelay *= 1000;
        }
        else if (arguments[idx] == "-j")
        {
            checkNextArgument();
            _jitter = parseInt();
            _jitter *= 1000;
            _jitterShift = _jitter / 2;
        }
    }
}

void Server::startSendChunk()
{
    MPI_Isend(_currentOutputBuffer.any, _bufferByteSize, MPI_BYTE, CLIENT_RANK, VALID_TAG, _clientServerComm, &_sendRequest);
}

void Server::startReceiveChunk()
{
    MPI_Irecv(_nextInputBuffer.any, _bufferByteSize, MPI_BYTE, CLIENT_RANK, MPI_ANY_TAG, _clientServerComm, &_receiveRequest);
}

bool Server::awaitSendChunk()
{
    MPI_Wait(&_sendRequest, MPI_STATUS_IGNORE);
    return true;
}

bool Server::awaitReceiveChunk()
{
    MPI_Status status;
    MPI_Wait(&_receiveRequest, &status);
    
    return status.MPI_TAG == VALID_TAG;
}

void Server::run()
{
    startReceiveChunk();
    startSendChunk();
    while (awaitReceiveChunk())
    {   
        swap(_currentInputBuffer.any, _nextInputBuffer.any);
        startReceiveChunk();

        processChunk();

        awaitSendChunk();
        swap(_currentOutputBuffer.any, _nextOutputBuffer.any);
        startSendChunk();
    }
}
