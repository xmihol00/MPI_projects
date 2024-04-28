#ifndef _CLIENT_H_
#define _CLIENT_H_

#include "client_server.h"
#include "portaudio.h"

#include <unistd.h>
#include <termios.h>

class Client : public ClientServer
{
public:
    Client(int argc, char **argv);
    ~Client();

    void run() override;

private:
    constexpr static int INITIAL_WRITE_PADDING{4};

    void parseArguments(int argc, char **argv) override;
    void startSendChunk() override;
    void startReceiveChunk() override;
    bool awaitSendChunk() override;
    bool awaitReceiveChunk() override;
    bool keyNotPressed();

    PaStream *_stream{nullptr};
    PaSampleFormat _samplingDatatype{paFloat32};

    timeval _terminalTimeout{tv_sec: 0, tv_usec: 0};
};

#endif