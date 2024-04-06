// =======================================================================================================================================================
// Algorithm:   Game of Life simulation
// Author:      David Mihola
// E-mail:      xmihol00@stud.fit.vutbr.cz
// Date:        1. 4. 2024
// Description: An implementation of Game of Life using the MPI library for communication. 
// =======================================================================================================================================================

#include <mpi.h>
#include <iostream>
#include <string>
#include <queue>
#include <bits/stdc++.h>
#include <array>

using namespace std;

#if 1
    #define INFO_PRINT_RANK0(rank, message) if (rank == 0) { cerr << "Info: " << message << endl; }
    #define INFO_PRINT(message) { cerr << "Info: " << message << endl; }
#else
    #define INFO_PRINT_RANK0(rank, message) 
    #define INFO_PRINT(message) { cerr << "Info: " << message << endl; }
#endif

class LifeSimulation
{
public:
    LifeSimulation(int argc, char **argv);
    ~LifeSimulation();
    void run();

private:
    void parseArguments(int argc, char **argv);
    void readInputFile();
    void initializeGridTopology();
    void initializeDataTypes();
    void initializeGrid();
    void performInitialScatter();
    void computeHaloZones();
    void startHaloZonesExchange();
    void computeTile();
    void awaitHaloZonesExchange();

    int _worldRank;
    int _worldSize;

    string _inputFileName;
    int _numberOfIterations;
    int _padding;

    MPI_Comm _subWorldComm;
    MPI_Comm _gridComm;

    MPI_Datatype _tileWithoutHaloZones;
    MPI_Datatype _tileWithoutHaloZonesResized;
    MPI_Datatype _tileWithHaloZones;

    MPI_Datatype _sendHaloZoneTypes[4];
    MPI_Datatype _recvHaloZoneTypes[4];

    struct Settings
    {
        int globalHeight;
        int globalWidth;
        int globalPaddedHeight;
        int globalPaddedWidth;
        int localHeight;
        int localWidth;
        int localHeightWithHaloZones;
        int localWidthWithHaloZones;
        int nodesHeightCount;
        int nodesWidthCount;
    } _settings;

    uint8_t *_globalTile = nullptr;
    uint8_t *_tiles[2] = {nullptr, nullptr};
};

LifeSimulation::LifeSimulation(int argc, char **argv)
{
    parseArguments(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &_worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &_worldSize);

    initializeGridTopology();
    readInputFile();

    /*initializeDataTypes();
    initializeGrid();
    performInitialScatter();*/
}

LifeSimulation::~LifeSimulation()
{
    if (_globalTile != nullptr)
    {
        free(_globalTile);
    }

    if (_tiles[0] != nullptr)
    {
        free(_tiles[0]);
    }

    if (_tiles[1] != nullptr)
    {
        free(_tiles[1]);
    }

    if (_subWorldComm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&_subWorldComm);
    }
}

void LifeSimulation::parseArguments(int argc, char **argv)
{
    _padding = 0;
    vector<string> arguments(argv, argv + argc);

    _inputFileName = arguments[1];
    try
    {
        _numberOfIterations = stoi(arguments[2]);
    }
    catch (const invalid_argument& e)
    {
        if (_worldRank == 0)
        {
            cerr << "Invalid 2nd argument (number of iterations): " << e.what() << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    } 
    catch (const out_of_range& e)
    {
        if (_worldRank == 0)
        {
            cerr << "Number of iterations is out of range: " << e.what() << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    if (_numberOfIterations < 0)
    {
        if (_worldRank == 0)
        {
            cerr << "Number of iterations must be a non-negative " << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (auto iterator = arguments.begin() + 3; iterator != arguments.end(); iterator++)
    {
        // TODO: parse other arguments
    }
}

void LifeSimulation::initializeGridTopology()
{
    int msbPosition = sizeof(int) * 8 - 1 - __builtin_clz(_worldSize);
    if (msbPosition & 1) // msb is odd
    {
        // count of nodes is different in each dimension
        _settings.nodesWidthCount = 1 << ((msbPosition >> 1) + 1); // twice as much nodes in the x direction
        _settings.nodesHeightCount = 1 << (msbPosition >> 1);
    }
    else
    {
        // count of nodes is same in both dimensions
        _settings.nodesWidthCount = 1 << (msbPosition >> 1);
        _settings.nodesHeightCount = 1 << (msbPosition >> 1);
    }

    if (msbPosition < 2 && _worldRank == 0)
    {
        cerr << "Error: The number of processes must be at least 4." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int color = _worldRank >= (1 << msbPosition); // exclude processes that will not fit on the grid
    if (color)
    {
        cerr << "Warning: Process " << _worldRank << " is not going to be part of the grid." << endl;
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, MPI_UNDEFINED, &_subWorldComm);
        _subWorldComm = MPI_COMM_NULL;
    }
    else
    {
        MPI_Comm_split(MPI_COMM_WORLD, color, _worldRank, &_subWorldComm);
    }

    MPI_Cart_create(_subWorldComm, 2, array<int, 2>{_settings.nodesHeightCount, _settings.nodesWidthCount}.data(), array<int, 2>{0, 0}.data(), 0, &_gridComm);
}

void LifeSimulation::readInputFile()
{
    if (_worldRank == 0) // only the root process reads the input file
    {
        ifstream inputFile(_inputFileName, ios::in);
        if (!inputFile.is_open())
        {
            cerr << "Error: Unable to open input file '" << _inputFileName << "'." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        inputFile.seekg(0, ios::end);
        int fileSize = inputFile.tellg();
        inputFile.seekg(0, ios::beg);

        string row;
        getline(inputFile, row);
        _settings.globalWidth = row.length();
        _settings.globalHeight = (fileSize + 1) / (_settings.globalWidth + 1); // +1 for the newline character
        INFO_PRINT_RANK0(_worldRank, "fileSize: " << fileSize << ", globalWidth: " << _settings.globalWidth << ", globalHeight: " << _settings.globalHeight)
        if (_settings.globalHeight * (_settings.globalWidth + 1) != fileSize && _settings.globalHeight * (_settings.globalWidth + 1) - 1 != fileSize)
        {
            cerr << "Error: The input file must contain a rectangular grid of '1' and '0' characters." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        _settings.globalPaddedWidth = ((_settings.globalWidth + _settings.nodesWidthCount - 1 + 2 * _padding) / _settings.nodesWidthCount) * _settings.nodesWidthCount;
        _settings.globalPaddedHeight = ((_settings.globalHeight + _settings.nodesHeightCount - 1  + 2 * _padding) / _settings.nodesHeightCount) * _settings.nodesHeightCount;
        _settings.localWidth = _settings.globalPaddedWidth / _settings.nodesWidthCount;
        _settings.localHeight = _settings.globalPaddedHeight / _settings.nodesHeightCount;
        _settings.localWidthWithHaloZones = _settings.localWidth + 2;
        _settings.localHeightWithHaloZones = _settings.localHeight + 2;
        int rowPadding = (_settings.globalPaddedWidth - _settings.globalWidth) >> 1;
        int colPadding = (_settings.globalPaddedHeight - _settings.globalHeight) >> 1;
        INFO_PRINT_RANK0(_worldRank, "globalPaddedWidth: " << _settings.globalPaddedWidth << ", globalPaddedHeight: " << _settings.globalPaddedHeight)

        int readLines = 0;
        _globalTile = static_cast<uint8_t *>(aligned_alloc(64, _settings.globalPaddedHeight * _settings.globalPaddedWidth * sizeof(uint8_t)));
        if (_globalTile == nullptr)
        {
            cerr << "Error: Unable to allocate memory for the global tile." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        memset(_globalTile, 0, _settings.globalPaddedHeight * _settings.globalPaddedWidth * sizeof(uint8_t));

        int idx = colPadding * _settings.globalPaddedWidth;
        do
        {
            idx += rowPadding;
            if (row.length() != static_cast<size_t>(_settings.globalWidth))
            {
                cerr << "Error: The input file must contain a rectangular grid of '1' and '0' characters." << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            readLines++;
            for (char &c : row)
            {
                _globalTile[idx++] = c == '1'; // consider all non-'1' characters as '0'
            }
            idx += rowPadding;
        }
        while (getline(inputFile, row));

        inputFile.close();
    }
}

void LifeSimulation::run()
{
    if (_worldRank == 0)
    {
        cerr << "Number of iterations: " << _numberOfIterations << endl;
        cerr << "globalHeight: " << _settings.globalHeight << endl;
        cerr << "globalWidth: " << _settings.globalWidth << endl;
        cerr << "globalPaddedHeight: " << _settings.globalPaddedHeight << endl;
        cerr << "globalPaddedWidth: " << _settings.globalPaddedWidth << endl;
        cerr << "localHeight: " << _settings.localHeight << endl;
        cerr << "localWidth: " << _settings.localWidth << endl;
        cerr << "localHeightWithHaloZones: " << _settings.localHeightWithHaloZones << endl;
        cerr << "localWidthWithHaloZones: " << _settings.localWidthWithHaloZones << endl;
        cerr << "nodesHeightCount: " << _settings.nodesHeightCount << endl;
        cerr << "nodesWidthCount: " << _settings.nodesWidthCount << endl;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    {
        LifeSimulation simulation(argc, argv);
        simulation.run();
    } // ensure all resources are freed before MPI_Finalize

    MPI_Finalize();
    return 0;
}
