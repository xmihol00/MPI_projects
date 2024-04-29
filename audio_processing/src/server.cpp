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

void Server::startSendChunk()
{
    MPI_Isend(_currentOutputBuffer, _bufferByteSize, MPI_BYTE, CLIENT_RANK, VALID_TAG, _clientServerComm, &_sendRequest);
}

void Server::startReceiveChunk()
{
    MPI_Irecv(_nextInputBuffer, _bufferByteSize, MPI_BYTE, CLIENT_RANK, MPI_ANY_TAG, _clientServerComm, &_receiveRequest);
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
    // initiate receiving and sending, the first sent chunk will be always empty
    startReceiveChunk();
    startSendChunk();

    while (awaitReceiveChunk()) // wait for new chunk to arrive
    {   
        swap(_currentInputBuffer, _nextInputBuffer);
        startReceiveChunk(); // start receiving the next chunk immediately to mask the processing delay

        processChunk();

        awaitSendChunk(); // wait for the previous chunk to finish
        swap(_currentOutputBuffer, _nextOutputBuffer);
        startSendChunk(); // start sending the newly processed chunk
    }
}
