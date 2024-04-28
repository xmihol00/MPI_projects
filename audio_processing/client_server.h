#ifndef _CLIENT_SERVER_H_
#define _CLIENT_SERVER_H_

#include <mpi.h>
#include <sndfile.hh>
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
    constexpr static int CLIENT_RANK{0};
    constexpr static int SERVER_RANK{1};
    constexpr static int ROOT{CLIENT_RANK};
    constexpr static int VALID_TAG{0};
    constexpr static int TERMINATING_TAG{1};

    virtual void parseArguments(int argc, char **argv);
    virtual void startSendChunk() = 0;
    virtual void startReceiveChunk() = 0;
    virtual bool awaitSendChunk() = 0;
    virtual bool awaitReceiveChunk() = 0;

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

    Buffer _currentInputBuffer;
    Buffer _currentOutputBuffer;
    Buffer _nextInputBuffer;
    Buffer _nextOutputBuffer;
    int _bufferByteSize{0};

    uint32_t _samplingRate{22050};
    uint32_t _samplesPerChunk{_samplingRate / 30};
    uint32_t _millisecondsPerChunk{0};
    uint32_t _channels{2};
};

#endif