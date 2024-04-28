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

    /**
     * @brief Runs the client or server side computation. The function is pure virtual and must be implemented by derived classes.
     */
    virtual void run() = 0;

protected:
    constexpr static int CLIENT_RANK{0};        ///< MPI rank (identifier) of the client process.
    constexpr static int SERVER_RANK{1};        ///< MPI rank (identifier) of the server process.
    constexpr static int ROOT{CLIENT_RANK};     ///< MPI rank of the root process.
    constexpr static int VALID_TAG{0};          ///< Tag used to indicate that the received chunk is valid.
    constexpr static int TERMINATING_TAG{1};    ///< Tag used to indicate that the received chunk is the last one.

    /**
     * @brief Parses the command line arguments.
     * 
     * @param argc The number of command line arguments.
     * @param argv The command line arguments.
     */
    virtual void parseArguments(int argc, char **argv);

    int _rank;                                      ///< MPI rank (identifier) of the current process.
    MPI_Comm _clientServerComm{MPI_COMM_NULL};      ///< MPI communicator for the client and server processes.

    MPI_Request _sendRequest{MPI_REQUEST_NULL};     ///< MPI request for sending a chunk of audio data.
    MPI_Request _receiveRequest{MPI_REQUEST_NULL};  ///< MPI request for receiving a chunk of audio data.

    // the buffers are continuously swapped between, i.e. a ping-pong buffering is used
    float *_currentInputBuffer;                     ///< Pointer to the current input buffer.
    float *_currentOutputBuffer;                    ///< Pointer to the current output buffer.
    float *_nextInputBuffer;                        ///< Pointer to the next input buffer.
    float *_nextOutputBuffer;                       ///< Pointer to the next output buffer;
    int _bufferByteSize{0};                         ///< Size of the input and output buffers in bytes.

    uint32_t _samplingRate{22050};                  ///< Default sampling rate of the audio data.
    uint32_t _samplesPerChunk{_samplingRate / 30};  ///< Default number of samples per chunk.
    uint32_t _millisecondsPerChunk{0};              ///< Number of milliseconds per chunk, 0 means use '_samplesPerChunk'.
    uint32_t _channels{2};                          ///< Default number of channels of the audio data.
};

#endif