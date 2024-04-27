#ifndef _CLIENT_H_
#define _CLIENT_H_

#include "client_server.h"
#include "portaudio.h"

#include <curses.h>  // sudo apt-get install libncurses5-dev libncursesw5-dev

class Client : public ClientServer
{
public:
    Client(int argc, char **argv);
    ~Client();

    void run() override;

private:
    constexpr static int INITIAL_WRITE_PADDING{4};
    constexpr static int SERVER_RANK{1};

    void parseArguments(int argc, char **argv) override;
    void startSendChunk() override;
    void startReceiveChunk() override;
    void awaitSendChunk() override;
    void awaitReceiveChunk() override;

    PaStream *_stream{nullptr};
    PaSampleFormat _samplingDatatype{paFloat32};
};

#endif