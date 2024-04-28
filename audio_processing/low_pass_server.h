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

    /**
     * @brief Parses the command line arguments.
     * 
     * @param argc The number of command line arguments.
     * @param argv The command line arguments.
     */
    void parseArguments(int argc, char **argv) override;

    /**
     * @brief Processes a chunk of audio data by applying a low pass filter.
     */
    void processChunk() override;

    uint32_t _processingDelay{0}; ///< Simulated processing delay in microseconds.
    uint32_t _jitter{1};          ///< Simulated jitter in microseconds.
    uint32_t _jitterShift{0};     ///< Jitter shift around 0 in microseconds.

    float _lastRightSample{0.0f};
    float _lastLeftSample{0.0f};
};

#endif