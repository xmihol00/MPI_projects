#ifndef _SERVER_H_
#define _SERVER_H_

#include <cstdlib>
#include "client_server.h"

class Server : public ClientServer
{
public:
    Server(int argc, char **argv);
    ~Server();

    /**
     * @brief Runs the server side computation.
     */
    void run() override;

protected:
    /**
     * @brief Processes a chunk of audio data. The functions is pure virtual and must be implemented by derived classes.
     */
    virtual void processChunk() = 0;

private:
    /**
     * @brief Starts sending a chunk of audio data to the client.
     */
    void startSendChunk();

    /**
     * @brief Starts receiving a chunk of audio data from the client.
     */
    void startReceiveChunk();

    /**
     * @brief Waits for the chunk of audio data to be sent to the client.
     * 
     * @return True if the chunk was sent successfully, false otherwise.
     */
    bool awaitSendChunk();

    /**
     * @brief Waits for the chunk of audio data to be received from the client.
     * 
     * @return True if the chunk was received successfully, false if it was the last chunk or a failure occurred.
     */
    bool awaitReceiveChunk();
};

#endif