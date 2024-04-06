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
    #define INFO_PRINT(message) { cerr << "Info rank " << _worldRank << ": " << message << endl; }
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
    void constructGridTopology();
    void destructGridTopology();
    void readInputFile();
    void constructDataTypes();
    void destructDataTypes();
    void constructGridTiles();
    void destructGridTiles();

    void performInitialDataExchange();
    void startCornersExchange(int sourceTile, int destinationTile);
    void awaitCornersExchange();
    void startHaloZonesExchange();
    void awaitHaloZonesExchange();
    void computeHaloZones();
    void computeTile();

    void printTilesWithHaloZones(int tile);
    void printGlobalTile();

    enum { NORTH = 0, SOUTH, WEST, EAST };
    enum { LEFT_UPPER = 0, RIGHT_UPPER, RIGHT_LOWER, LEFT_LOWER };

    MPI_Comm _subWorldCommunicator;
    MPI_Comm _gridCommunicator;

    int _worldRank;
    int _worldSize;
    int _gridSize;
    int _gridRank;

    struct Arguments
    {
        int numberOfIterations;
        int padding;
        int wraparound;
        string inputFileName;
    } _arguments;

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
        int localTileSize;
        int localTileSizeWithHaloZones;
        int nodesHeightCount;
        int nodesWidthCount;
        int nodesTotalCount;
    } _settings;

    MPI_Datatype _tileWithoutHaloZones;
    MPI_Datatype _tileWithoutHaloZonesResized;
    MPI_Datatype _tileWithHaloZones;

    MPI_Datatype _sendHaloZoneTypes[4];
    MPI_Datatype _recvHaloZoneTypes[4];

    vector<int> _scatterGatherCounts;
    vector<int> _scatterGatherDisplacements;

    int _neighbors[4];
    int _cornerNeighbors[4];
    MPI_Request _cornerSendRequests[4];
    MPI_Request _cornerRecvRequests[4];
    int _neighbourCounts[4] = {1, 1, 1, 1};
    MPI_Aint _neighbourDisplacements[4] = {0, 0, 0, 0};

    uint8_t *_globalTile = nullptr;
    uint8_t *_tiles[2] = {nullptr, nullptr}; // two tiles for ping-pong computation

    inline constexpr bool isActiveProcess() { return _subWorldCommunicator != MPI_COMM_NULL; };

    inline constexpr bool isNotTopRow() { return _neighbors[NORTH] != MPI_PROC_NULL; };
    inline constexpr bool isNotBottomRow() { return _neighbors[SOUTH] != MPI_PROC_NULL; };
    inline constexpr bool isNotLeftColumn() { return _neighbors[WEST] != MPI_PROC_NULL; };
    inline constexpr bool isNotRightColumn() { return _neighbors[EAST] != MPI_PROC_NULL; };
};

LifeSimulation::LifeSimulation(int argc, char **argv)
{
    parseArguments(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &_worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &_worldSize);

    constructGridTopology();
    readInputFile();

    if (isActiveProcess())
    {
        constructDataTypes();
        constructGridTiles();
    }
}

LifeSimulation::~LifeSimulation()
{
    if (isActiveProcess())
    {
        destructGridTiles();
        destructDataTypes();
        destructGridTopology();
    }
}

void LifeSimulation::parseArguments(int argc, char **argv)
{
    _arguments.padding = 0;
    _arguments.wraparound = 0;
    vector<string> arguments(argv, argv + argc);

    _arguments.inputFileName = arguments[1];
    try
    {
        _arguments.numberOfIterations = stoi(arguments[2]);
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
    
    if (_arguments.numberOfIterations < 0)
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

void LifeSimulation::constructGridTopology()
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
    _settings.nodesTotalCount = _settings.nodesWidthCount * _settings.nodesHeightCount;

    if (msbPosition < 2 && _worldRank == 0)
    {
        cerr << "Error: The number of processes must be at least 4." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int color = _worldRank >= (1 << msbPosition); // exclude processes that will not fit on the grid
    if (color)
    {
        cerr << "Warning: Process " << _worldRank << " is not going to be part of the grid." << endl;
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, MPI_UNDEFINED, &_subWorldCommunicator);
        _subWorldCommunicator = MPI_COMM_NULL;
        _gridCommunicator = MPI_COMM_NULL;

        return; // this process will not be part of the grid, therefore it will be idle
    }
    else
    {
        MPI_Comm_split(MPI_COMM_WORLD, color, _worldRank, &_subWorldCommunicator);
    }

    MPI_Cart_create(_subWorldCommunicator, 2, 
                    array<int, 2>{_settings.nodesHeightCount, _settings.nodesWidthCount}.data(), 
                    array<int, 2>{_arguments.wraparound, _arguments.wraparound}.data(), 
                    0, &_gridCommunicator);
    MPI_Comm_rank(_gridCommunicator, &_gridRank);
    MPI_Comm_size(_gridCommunicator, &_gridSize);

    MPI_Cart_shift(_gridCommunicator, 0, 1, &_neighbors[WEST], &_neighbors[EAST]);
    MPI_Cart_shift(_gridCommunicator, 1, 1, &_neighbors[NORTH], &_neighbors[SOUTH]);

    int coordinates[2];
    MPI_Cart_coords(_gridCommunicator, _gridRank, 2, coordinates);

    // TODO: adjust for wraparound
    // left top corner
    coordinates[0]--;
    coordinates[1]--;
    if (coordinates[0] >= 0 && coordinates[1] >= 0)
    {
        MPI_Cart_rank(_gridCommunicator, coordinates, &_cornerNeighbors[0]);
    }
    else
    {
        _cornerNeighbors[LEFT_UPPER] = MPI_PROC_NULL;
    }

    // right top corner
    coordinates[1] += 2;
    if (coordinates[0] >= 0 && coordinates[1] < _settings.nodesWidthCount)
    {
        MPI_Cart_rank(_gridCommunicator, coordinates, &_cornerNeighbors[1]);
    }
    else
    {
        _cornerNeighbors[RIGHT_UPPER] = MPI_PROC_NULL;
    }

    // right bottom corner
    coordinates[0] += 2;
    if (coordinates[0] < _settings.nodesHeightCount && coordinates[1] < _settings.nodesWidthCount)
    {
        MPI_Cart_rank(_gridCommunicator, coordinates, &_cornerNeighbors[2]);
    }
    else
    {
        _cornerNeighbors[RIGHT_LOWER] = MPI_PROC_NULL;
    }

    // left bottom corner
    coordinates[1] -= 2;
    if (coordinates[0] < _settings.nodesHeightCount && coordinates[1] >= 0)
    {
        MPI_Cart_rank(_gridCommunicator, coordinates, &_cornerNeighbors[3]);
    }
    else
    {
        _cornerNeighbors[LEFT_LOWER] = MPI_PROC_NULL;
    }
}

void LifeSimulation::destructGridTopology()
{
    if (_subWorldCommunicator != MPI_COMM_NULL)
    {
        MPI_Comm_free(&_subWorldCommunicator);
    }

    if (_gridCommunicator != MPI_COMM_NULL)
    {
        MPI_Comm_free(&_gridCommunicator);
    }
}

void LifeSimulation::readInputFile()
{
    if (_worldRank == 0) // only the root process reads the input file
    {
        ifstream inputFile(_arguments.inputFileName, ios::in);
        if (!inputFile.is_open())
        {
            cerr << "Error: Unable to open input file '" << _arguments.inputFileName << "'." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        inputFile.seekg(0, ios::end);
        int fileSize = inputFile.tellg();
        inputFile.seekg(0, ios::beg);

        string row;
        getline(inputFile, row);
        _settings.globalWidth = row.length();
        _settings.globalHeight = (fileSize + 1) / (_settings.globalWidth + 1); // +1 for the newline character
        if (_settings.globalHeight * (_settings.globalWidth + 1) != fileSize && _settings.globalHeight * (_settings.globalWidth + 1) - 1 != fileSize)
        {
            cerr << "Error: The input file must contain a rectangular grid of '1' and '0' characters." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        _settings.globalPaddedWidth = ((_settings.globalWidth + _settings.nodesWidthCount - 1 + 2 * _arguments.padding) / _settings.nodesWidthCount) * _settings.nodesWidthCount;
        _settings.globalPaddedHeight = ((_settings.globalHeight + _settings.nodesHeightCount - 1  + 2 * _arguments.padding) / _settings.nodesHeightCount) * _settings.nodesHeightCount;
        _settings.localWidth = _settings.globalPaddedWidth / _settings.nodesWidthCount;
        _settings.localHeight = _settings.globalPaddedHeight / _settings.nodesHeightCount;
        _settings.localWidthWithHaloZones = _settings.localWidth + 2;
        _settings.localHeightWithHaloZones = _settings.localHeight + 2;
        _settings.localTileSize = _settings.localHeight * _settings.localWidth;
        _settings.localTileSizeWithHaloZones = _settings.localHeightWithHaloZones * _settings.localWidthWithHaloZones;
        int rowPadding = (_settings.globalPaddedWidth - _settings.globalWidth) >> 1;
        int colPadding = (_settings.globalPaddedHeight - _settings.globalHeight) >> 1;

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

    // broadcast the additionally obtained settings to all processes
    MPI_Bcast(&_settings, sizeof(Settings), MPI_BYTE, 0, MPI_COMM_WORLD);
}

void LifeSimulation::constructDataTypes()
{
    int outerTileSizes[2] = {_settings.globalPaddedHeight, _settings.globalPaddedWidth};
    int innerTileSizes[2] = {_settings.localHeight, _settings.localWidth};
    int starts[2] = {0, 0};
    
    // tile data type for scatter and gather data type
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_BYTE, &_tileWithoutHaloZones);
    MPI_Type_commit(&_tileWithoutHaloZones);
    MPI_Type_create_resized(_tileWithoutHaloZones, 0, _settings.localWidth * sizeof(uint8_t), &_tileWithoutHaloZonesResized);
    MPI_Type_commit(&_tileWithoutHaloZonesResized);

    // tile data type with halo zones
    outerTileSizes[0] = _settings.localHeightWithHaloZones;
    outerTileSizes[1] = _settings.localWidthWithHaloZones;
    starts[0] = 1;
    starts[1] = 1;
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_BYTE, &_tileWithHaloZones);
    MPI_Type_commit(&_tileWithHaloZones);
    
    // starts and displacements of each tile for scatter and gather
    _scatterGatherCounts.resize(_settings.nodesTotalCount);
    _scatterGatherDisplacements.resize(_settings.nodesTotalCount);
    for (int i = 0; i < _settings.nodesTotalCount; i++)
    {
        _scatterGatherCounts[i] = 1;
        // ensure displacements are increasing by 1 on the same row of nodes and between rows of nodes by the multiple of local tile hight and width
        // i.e. each row of nodes must displace itself by multiple of whole tiles, while within the row of nodes the displacement is by 1
        _scatterGatherDisplacements[i] = (i % _settings.nodesWidthCount) + _settings.localHeight * (i / _settings.nodesWidthCount) * _settings.nodesWidthCount;
    }

    // halo zone data types
    innerTileSizes[0] = 1;
    innerTileSizes[1] = _settings.localWidth;

    // north halo zone for neighbour all to all exchange
    starts[0] = 1;
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_BYTE, &_sendHaloZoneTypes[NORTH]);
    MPI_Type_commit(&_sendHaloZoneTypes[NORTH]);
    starts[0] = 0;
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_BYTE, &_recvHaloZoneTypes[NORTH]);
    MPI_Type_commit(&_recvHaloZoneTypes[NORTH]);

    // south halo zone for neighbour all to all exchange
    starts[0] = _settings.localHeight;
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_BYTE, &_sendHaloZoneTypes[SOUTH]);
    MPI_Type_commit(&_sendHaloZoneTypes[SOUTH]);
    starts[0] = _settings.localHeight + 1;
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_BYTE, &_recvHaloZoneTypes[SOUTH]);
    MPI_Type_commit(&_recvHaloZoneTypes[SOUTH]);

    innerTileSizes[0] = _settings.localHeight;
    innerTileSizes[1] = 1;
    starts[0] = 1;

    // west halo zone for neighbour all to all exchange
    starts[1] = 1;
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_BYTE, &_sendHaloZoneTypes[WEST]);
    MPI_Type_commit(&_sendHaloZoneTypes[WEST]);
    starts[1] = 0;
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_BYTE, &_recvHaloZoneTypes[WEST]);
    MPI_Type_commit(&_recvHaloZoneTypes[WEST]);

    // east halo zone for neighbour all to all exchange
    starts[1] = _settings.localWidth;
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_BYTE, &_sendHaloZoneTypes[EAST]);
    MPI_Type_commit(&_sendHaloZoneTypes[EAST]);
    starts[1] = _settings.localWidth + 1;
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_BYTE, &_recvHaloZoneTypes[EAST]);
    MPI_Type_commit(&_recvHaloZoneTypes[EAST]);
}

void LifeSimulation::destructDataTypes()
{
    MPI_Type_free(&_tileWithoutHaloZones);
    MPI_Type_free(&_tileWithoutHaloZonesResized);
    MPI_Type_free(&_tileWithHaloZones);

    MPI_Type_free(&_sendHaloZoneTypes[NORTH]);
    MPI_Type_free(&_sendHaloZoneTypes[SOUTH]);
    MPI_Type_free(&_sendHaloZoneTypes[WEST]);
    MPI_Type_free(&_sendHaloZoneTypes[EAST]);

    MPI_Type_free(&_recvHaloZoneTypes[NORTH]);
    MPI_Type_free(&_recvHaloZoneTypes[SOUTH]);
    MPI_Type_free(&_recvHaloZoneTypes[WEST]);
    MPI_Type_free(&_recvHaloZoneTypes[EAST]);
}

void LifeSimulation::constructGridTiles()
{
    _tiles[0] = static_cast<uint8_t *>(aligned_alloc(64, _settings.localTileSizeWithHaloZones * sizeof(uint8_t)));
    _tiles[1] = static_cast<uint8_t *>(aligned_alloc(64, _settings.localTileSizeWithHaloZones * sizeof(uint8_t)));
    if (_tiles[0] == nullptr || _tiles[1] == nullptr)
    {
        cerr << "Error: Unable to allocate memory for local tiles." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void LifeSimulation::destructGridTiles()
{
    if (_tiles[0] != nullptr)
    {
        free(_tiles[0]);
    }

    if (_tiles[1] != nullptr)
    {
        free(_tiles[1]);
    }
}

void LifeSimulation::performInitialDataExchange()
{
    // partition the global tile into local tiles and scatter them to all processes in the grid
    MPI_Scatterv(_globalTile, _scatterGatherCounts.data(), _scatterGatherDisplacements.data(), _tileWithoutHaloZonesResized, 
                 _tiles[0], 1, _tileWithHaloZones, 0, _gridCommunicator);
    // perform the initial exchange of halo zones
    MPI_Neighbor_alltoallw(_tiles[0], _neighbourCounts, _neighbourDisplacements, _sendHaloZoneTypes, 
                           _tiles[0], _neighbourCounts, _neighbourDisplacements, _recvHaloZoneTypes, _gridCommunicator);
    // perform the initial exchange of corner points
    startCornersExchange(0, 0);
    awaitCornersExchange();
    // copy the initial data into the second tile
    copy(_tiles[0], _tiles[0] + _settings.localTileSize, _tiles[1]);
}

void LifeSimulation::startCornersExchange(int sourceTile, int destinationTile)
{
    if (_cornerNeighbors[LEFT_UPPER] != MPI_PROC_NULL)
    {
        MPI_Isend(_tiles[sourceTile] + _settings.localWidthWithHaloZones + 1, 1, MPI_BYTE, _cornerNeighbors[LEFT_UPPER], 0, 
                  _gridCommunicator, &_cornerSendRequests[LEFT_UPPER]);
        MPI_Irecv(_tiles[destinationTile], 1, MPI_BYTE, _cornerNeighbors[LEFT_UPPER], 0, 
                  _gridCommunicator, &_cornerRecvRequests[LEFT_UPPER]);
    }

    if (_cornerNeighbors[RIGHT_UPPER] != MPI_PROC_NULL)
    {
        MPI_Isend(_tiles[sourceTile] +  2 * _settings.localWidthWithHaloZones - 2, 1, MPI_BYTE, _cornerNeighbors[RIGHT_UPPER], 0, 
                  _gridCommunicator, &_cornerSendRequests[RIGHT_UPPER]);
        MPI_Irecv(_tiles[destinationTile] + _settings.localWidthWithHaloZones - 1, 1, MPI_BYTE, _cornerNeighbors[RIGHT_UPPER], 0, 
                  _gridCommunicator, &_cornerRecvRequests[RIGHT_UPPER]);
    }

    if (_cornerNeighbors[RIGHT_LOWER] != MPI_PROC_NULL)
    {
        MPI_Isend(_tiles[sourceTile] + _settings.localTileSizeWithHaloZones - _settings.localWidthWithHaloZones - 2, 1, MPI_BYTE, _cornerNeighbors[RIGHT_LOWER], 0, 
                  _gridCommunicator, &_cornerSendRequests[RIGHT_LOWER]);
        MPI_Irecv(_tiles[destinationTile] + _settings.localTileSizeWithHaloZones - 1, 1, MPI_BYTE, _cornerNeighbors[RIGHT_LOWER], 0,
                  _gridCommunicator, &_cornerRecvRequests[RIGHT_LOWER]);
    }

    if (_cornerNeighbors[LEFT_LOWER] != MPI_PROC_NULL)
    {
        MPI_Isend(_tiles[sourceTile] + _settings.localWidthWithHaloZones - 2 * _settings.localWidthWithHaloZones + 1, 1, MPI_BYTE, _cornerNeighbors[LEFT_LOWER], 0, 
                  _gridCommunicator, &_cornerSendRequests[LEFT_LOWER]);
        MPI_Irecv(_tiles[destinationTile] + _settings.localTileSizeWithHaloZones - _settings.localWidthWithHaloZones, 1, MPI_BYTE, _cornerNeighbors[LEFT_LOWER], 0,
                  _gridCommunicator, &_cornerRecvRequests[LEFT_LOWER]);
    }
}

void LifeSimulation::awaitCornersExchange()
{
    if (_cornerNeighbors[LEFT_UPPER] != MPI_PROC_NULL)
    {
        MPI_Wait(&_cornerSendRequests[LEFT_UPPER], MPI_STATUS_IGNORE);
        MPI_Wait(&_cornerRecvRequests[LEFT_UPPER], MPI_STATUS_IGNORE);
    }

    if (_cornerNeighbors[RIGHT_UPPER] != MPI_PROC_NULL)
    {
        MPI_Wait(&_cornerSendRequests[RIGHT_UPPER], MPI_STATUS_IGNORE);
        MPI_Wait(&_cornerRecvRequests[RIGHT_UPPER], MPI_STATUS_IGNORE);
    }

    if (_cornerNeighbors[RIGHT_LOWER] != MPI_PROC_NULL)
    {
        MPI_Wait(&_cornerSendRequests[RIGHT_LOWER], MPI_STATUS_IGNORE);
        MPI_Wait(&_cornerRecvRequests[RIGHT_LOWER], MPI_STATUS_IGNORE);
    }

    if (_cornerNeighbors[LEFT_LOWER] != MPI_PROC_NULL)
    {
        MPI_Wait(&_cornerSendRequests[LEFT_LOWER], MPI_STATUS_IGNORE);
        MPI_Wait(&_cornerRecvRequests[LEFT_LOWER], MPI_STATUS_IGNORE);
    }
}

void LifeSimulation::printTilesWithHaloZones(int tile)
{
    for (int node = 0; node < _gridSize; node++)
    {
        if (_gridRank == node)
        {
            cerr << "Rank: " << _gridRank << "\n";
            for (int i = 0; i < _settings.localHeightWithHaloZones; i++)
            {
                for (int j = 0; j < _settings.localWidthWithHaloZones; j++)
                {
                    if (j == 1 || j == _settings.localWidthWithHaloZones - 1) // separate west and east halo zones
                    {
                        cerr << " ";
                    }
                    cerr << static_cast<int>(_tiles[tile][i * _settings.localWidthWithHaloZones + j]);
                }
                cerr << "\n";
                
                if (i == 0 || i == _settings.localHeightWithHaloZones - 2) // separate north and south halo zones
                {
                    cerr << "\n";
                }
            }
        }
        cerr << endl;
        MPI_Barrier(_gridCommunicator);
    }
}

void LifeSimulation::printGlobalTile()
{
    if (_worldRank == 0)
    {
        cerr << "Global tile:\n";
        for (int i = 0; i < _settings.globalPaddedHeight; i++)
        {
            for (int j = 0; j < _settings.globalPaddedWidth; j++)
            {
                cerr << static_cast<int>(_globalTile[i * _settings.globalPaddedWidth + j]);
            }
            cerr << "\n";
        }
        cerr << endl;
    }
}

void LifeSimulation::run()
{
    printGlobalTile();

    if (_worldRank == 0)
    {
        cerr << "Number of iterations: " << _arguments.numberOfIterations << endl;
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
        cerr << "nodesTotalCount: " << _settings.nodesTotalCount << endl;
    }

    performInitialDataExchange();
    printTilesWithHaloZones(0);
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
