#ifndef _LOW_PASS_SERVER_H_
#define _LOW_PASS_SERVER_H_

#include "server.h"

#include <unistd.h>

class LowPassServer : public Server
{
public:
    LowPassServer(int argc, char **argv);
    ~LowPassServer();

private:
    constexpr static uint32_t EXPECTED_CHANNELS{2};
    constexpr static float LOW_PASS_COEFFICIENT{0.05f};

    void processChunk() override;

    float _lastRightSample{0.0f};
    float _lastLeftSample{0.0f};
};

#endif