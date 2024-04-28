#ifndef _SERVER_H_
#define _SERVER_H_

#include <cstdlib>
#include "client_server.h"

class Server : public ClientServer
{
public:
    Server(int argc, char **argv);
    ~Server();

    void run() override;

protected:
    virtual void processChunk() = 0;
    void parseArguments(int argc, char **argv) override;

    uint32_t _processingDelay{0};
    uint32_t _jitter{1};
    uint32_t _jitterShift{0};

private:
    void startSendChunk() override;
    void startReceiveChunk() override;
    bool awaitSendChunk() override;
    bool awaitReceiveChunk() override;
};

#endif