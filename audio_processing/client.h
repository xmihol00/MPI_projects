#ifndef _CLIENT_H_
#define _CLIENT_H_

#include "client_server.h"

#include <portaudio.h>
#include <sndfile.hh>
#include <unistd.h>
#include <omp.h>
#include <chrono>
#include <thread>

class Client : public ClientServer
{
public:
    Client(int argc, char **argv);
    ~Client();

    /**
     * @brief Runs the client side computation.
     */
    void run() override;

private:
    /**
     * @brief Parses the command line arguments.
     * 
     * @param argc The number of command line arguments.
     * @param argv The command line arguments.
     */
    void parseArguments(int argc, char **argv) override;

    /**
     * @brief Sends a chunk of audio data to the server.
     */
    void sendChunk();

    /**
     * @brief Receives a chunk of audio data from the server.
     */
    void receiveChunk();

    /**
     * @brief Starts sending a chunk of audio data to the server.
     */
    void startSendChunk();

    /**
     * @brief Starts receiving a chunk of audio data from the server.
     */
    void startReceiveChunk();

    /**
     * @brief Waits for the chunk of audio data to be sent to the server.
     * 
     * @return True if the chunk was sent successfully, false otherwise.
     */
    bool awaitSendChunk();

    /**
     * @brief Waits for the chunk of audio data to be received from the server.
     * 
     * @return True if the chunk was received successfully, false otherwise.
     */
    bool awaitReceiveChunk();

    /**
     * @brief Polls the keyboard for the 'Enter' key press.
     * 
     * @return False if the 'Enter' key was pressed, true otherwise.
     */
    bool enterNotPressed();

    /**
     * @brief Sends a terminating message to the server.
     */
    void stopProcessing();

    PaStream *_stream{nullptr};          ///< PortAudio stream to record audio from and write the processed audio back. 

    uint32_t _numberOfUsedBuffers{2};    ///< Number of buffers used for audio processing.
    std::vector<float *> _outputBuffers; ///< Output buffers for audio data.
    float  *_inputBuffer;                ///< Input buffer for audio data.
    uint32_t _outputBufferWriteIdx{0};   ///< Index of the output buffer for writing.
    uint32_t _outputBufferReadIdx{0};    ///< Index of the output buffer for reading.
    bool _noSimulation{false};           ///< Flag indicating that simulation is disabled and audio data are processed as fast as possible.

    std::string _inputFileName;          ///< Name of the input audio file.
    std::string _outputFileName;         ///< Name of the output audio file.
    SndfileHandle _inputFile;            ///< Handle to the input audio file.
    SndfileHandle _outputFile;           ///< Handle to the output audio file.
};

#endif