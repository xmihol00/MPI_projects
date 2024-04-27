#include "server.h"

Server::Server(int argc, char **argv) : ClientServer(argc, argv)
{
    allocateBuffers();
}

Server::~Server()
{
    freeBuffers();
}

void Server::startSendChunk()
{
    MPI_Isend(_outputBuffer.any, _bufferByteSize, MPI_BYTE, CLIENT_RANK, 0, MPI_COMM_WORLD, &_sendRequest);
}

void Server::startReceiveChunk()
{
    MPI_Irecv(_inputBuffer.any, _bufferByteSize, MPI_BYTE, CLIENT_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, &_receiveRequest);
}

void Server::awaitSendChunk()
{
    MPI_Wait(&_sendRequest, MPI_STATUS_IGNORE);
}

void Server::awaitReceiveChunk()
{
    MPI_Status status;
    MPI_Wait(&_receiveRequest, &status);
    
    _run = status.MPI_TAG == 0;
}

void Server::run()
{
    startReceiveChunk();
    startSendChunk();
    while (_run)
    {
        awaitReceiveChunk();
        processChunk();
        startReceiveChunk();

        awaitSendChunk();
        startSendChunk();
    }
}
