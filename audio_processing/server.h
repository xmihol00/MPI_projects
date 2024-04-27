#ifndef _SERVER_H_
#define _SERVER_H_

#include "client_server.h"

class Server : public ClientServer
{
public:
    Server(int argc, char **argv);
    ~Server();

    void run() override;

protected:
    constexpr static int CLIENT_RANK{0};

    void parseArguments(int argc, char **argv) override;
    void startSendChunk() override;
    void startReceiveChunk() override;
    void awaitSendChunk() override;
    void awaitReceiveChunk() override;

    virtual void allocateBuffers() = 0;
    virtual void freeBuffers() = 0;
    virtual void processChunk() = 0;

    bool _run = true;
};

#endif