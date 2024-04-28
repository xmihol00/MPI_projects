#include "low_pass_server.h"

using namespace std;

LowPassServer::LowPassServer(int argc, char **argv) : Server(argc, argv)
{
    if (_channels != EXPECTED_CHANNELS)
    {
        cerr << "Low pass server expects " << EXPECTED_CHANNELS << " channels, but received " << _channels << " channels." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    _currentInputBuffer.f32 = new float[_samplesPerChunk * _channels];
    _currentOutputBuffer.f32 = new float[_samplesPerChunk * _channels];
    _nextInputBuffer.f32 = new float[_samplesPerChunk * _channels];
    _nextOutputBuffer.f32 = new float[_samplesPerChunk * _channels];
    _bufferByteSize = _samplesPerChunk * _channels * sizeof(float);
}

LowPassServer::~LowPassServer()
{
    delete[] _nextOutputBuffer.f32;
    delete[] _nextInputBuffer.f32;
    delete[] _currentOutputBuffer.f32;
    delete[] _currentInputBuffer.f32;

    Server::~Server();
}

void LowPassServer::processChunk()
{
    for (uint32_t i = 0; i < _samplesPerChunk * _channels; i += _channels)
    {
        _lastRightSample = _lastRightSample - (LOW_PASS_COEFFICIENT * (_lastRightSample - _currentInputBuffer.f32[i]));
        _lastLeftSample = _lastLeftSample - (LOW_PASS_COEFFICIENT * (_lastLeftSample - _currentInputBuffer.f32[i + 1]));

        _currentOutputBuffer.f32[i] = _lastRightSample;
        _currentOutputBuffer.f32[i + 1] = _lastLeftSample;
    }   
}
