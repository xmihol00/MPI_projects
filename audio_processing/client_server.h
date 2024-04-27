#ifndef _CLIENT_SERVER_H_
#define _CLIENT_SERVER_H_

#include "mpi.h"
#include <vector>
#include <string>
#include <iostream>

class ClientServer
{
public:
    ClientServer(int argc, char **argv);
    ~ClientServer();
    virtual void run() = 0;

protected:
    virtual void parseArguments(int argc, char **argv);
    virtual void startSendChunk() = 0;
    virtual void startReceiveChunk() = 0;
    virtual void awaitSendChunk() = 0;
    virtual void awaitReceiveChunk() = 0;

    constexpr static int ROOT{0};
    int _rank;
    MPI_Comm _clientServerComm{MPI_COMM_NULL};

    MPI_Request _sendRequest{MPI_REQUEST_NULL};
    MPI_Request _receiveRequest{MPI_REQUEST_NULL};

    union Buffer
    {
        float   *f32;
        int32_t *i32;
        int16_t *i16;
        int8_t  *i8;
        uint8_t *u8;
        void    *any;
    };

    Buffer _inputBuffer;
    Buffer _outputBuffer;
    int _bufferByteSize{0};

    uint32_t _samplingRate{44100};
    uint32_t _samplesPerChunk{_samplingRate / 10};
    uint32_t _millisecondsPerChunk{0};
    uint32_t _channels{2};
};

#endif