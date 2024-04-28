#include "server.h"

using namespace std;

Server::Server(int argc, char **argv) : ClientServer(argc, argv) 
{ 

}

Server::~Server()
{
    ClientServer::~ClientServer();
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
