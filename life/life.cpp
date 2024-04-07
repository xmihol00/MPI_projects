// =======================================================================================================================================================
// Algorithm:   Game of Life simulation
// Author:      David Mihola
// E-mail:      xmihol00@stud.fit.vutbr.cz
// Date:        1. 4. 2024
// Description: An implementation of Game of Life using the MPI library for communication. 
// =======================================================================================================================================================

// =======================================================================================================================================================
// Description of the solution:
// 1. Processes are organized into a 2D (cartesian) mesh topology. The dimensions of the mesh can be specified by the user, see TODO, or are derived 
//    automatically. The automatically derived mesh dimensions will always contain a power of 2 processes, see TODO. Processes that do not fit into the 
//    mesh will not be utilized. Lastly, each process in the mesh retrieves ranks of its neighbors, especially of its corner neighbors (NW, NE, SE, SW).
// 2. The input file is read only by the root process. Based on the size of the input file and the length of the first row, the global grid dimensions, 
//    i.e. the simulation space, are determined. The global grid dimensions are adjusted to be divisible by the number of nodes in each dimension of the
//    mesh. The content of the file is then read to the global grid with evenly distributed padding of '0' if necessary or as specified by the user.
// 3. The root process computes the sizes of local tiles, i.e. the parts of the global grid assigned to each process, and broadcasts this information to
//    all the processes.
// 4. Each process creates MPI data types specifying its local tile, the tile with halo zones (edges of the local tiles of neighboring processes, which 
//    must be visible to a given process), and the halo zones. These MPI data types are same for all processes. 
// 5. Each process allocates memory for two local tiles, one for the current state of the simulation and one for the next state to realize so called 
//    ping-pong buffering.
// 6. The root process partitions (not explicitly, this is done by MPI) the global grid into local tiles and scatters them to all processes in the mesh.
//    Initial halo zone exchange is performed, i.e. each process sends its halo zones to its neighbors and receives halo zones from its neighbors. Now, 
//    all processes have the initial state, which is also copied to the second local tile.
// 7. The simulation is started. In each iteration, the following steps are performed:
//    a) Each process computes the next state around the edges of its local tile, i.e. the halo zones.
//    b) Non-blocking exchange of halo zones is initiated.
//    c) The computation of the next state is performed (tile edges not included anymore). This is accelerated with SIMD instructions.
//    d) The exchange of halo zones is awaited.
//    e) current and next state tiles are swapped.
// 8. After the simulation is finished, the local tiles are gathered back to the root process (without the halo zones) and the content of the global grid
//    is printed in a table-like format which shows what part of the grid was computed by which process.
// =======================================================================================================================================================

#include <mpi.h>
#include <iostream>
#include <string>
#include <queue>
#include <bits/stdc++.h>
#include <array>

using namespace std;
using cell_t = uint8_t;

#if 0
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
    void constructMeshTopology();
    void destructMeshTopology();
    void readInputFile();
    void constructDataTypes();
    void destructDataTypes();
    void constructGridTiles();
    void destructGridTiles();

    void exchangeInitialData();
    void startHaloZonesExchange();
    void awaitHaloZonesExchange();
    void computeHaloZones();
    void computeTile();
    void collectResults();

    #pragma omp declare simd
    inline constexpr cell_t updateCell
    (
        cell_t leftUpperCorner, cell_t upper, cell_t rightUpperCorner, 
        cell_t left, cell_t center, cell_t right, 
        cell_t leftLowerCorner, cell_t lower, cell_t rightLowerCorner
    )
    {
        int sum = leftUpperCorner + upper + rightUpperCorner + left + right + leftLowerCorner + lower + rightLowerCorner;
        return (center && ((sum == 2) || (sum == 3))) || (!center && (sum == 3));
    }

    void debugPrintLocalTile(bool current);
    void testPrintGlobalTile();
    void prettyPrintGlobalTile();

    enum { NORTH = 0, SOUTH, WEST, EAST };
    enum { LEFT_UPPER = 0, RIGHT_UPPER, RIGHT_LOWER, LEFT_LOWER };
    static constexpr int ROOT = 0;

    MPI_Comm _subWorldCommunicator;
    MPI_Comm _meshCommunicator;

    int _worldRank;
    int _worldSize;
    int _meshSize;
    int _meshRank;

    struct Arguments
    {
        string inputFileName;
        int numberOfIterations;
        int padding = 0;
        int paddingHeight = 0;
        int paddingWidth = 0;
        bool wraparound = false;
        int nodesHeightCount = 0;
        int nodesWidthCount = 0;
    } _arguments;

    struct Settings
    {
        int globalHeight;
        int globalWidth;
        int globalNotPaddedHeight;
        int globalNotPaddedWidth;
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

    MPI_Datatype _tileType;
    MPI_Datatype _tileResizedType;
    MPI_Datatype _tileWithHaloZonesType;

    MPI_Datatype _sendHaloZoneTypes[4];
    MPI_Datatype _recvHaloZoneTypes[4];

    vector<int> _scatterGatherCounts;
    vector<int> _scatterGatherDisplacements;

    MPI_Request _haloZoneRequest;

    int _neighbors[4];
    int _cornerNeighbors[4];
    MPI_Request _cornerSendRequests[4];
    MPI_Request _cornerRecvRequests[4];
    int _neighbourCounts[4] = {1, 1, 1, 1};
    MPI_Aint _neighbourDisplacements[4] = {0, 0, 0, 0};

    cell_t *_globalTile = nullptr;
    // two tiles for ping-pong buffering and to overlap computation with communication
    cell_t *_currentTile = nullptr;
    cell_t *_nextTile = nullptr;

    inline constexpr bool isActiveProcess() { return _subWorldCommunicator != MPI_COMM_NULL; };

    inline constexpr bool isTopRow()      { return _neighbors[NORTH] == MPI_PROC_NULL; };
    inline constexpr bool isBottomRow()   { return _neighbors[SOUTH] == MPI_PROC_NULL; };
    inline constexpr bool isLeftColumn()  { return _neighbors[WEST]  == MPI_PROC_NULL; };
    inline constexpr bool isRightColumn() { return _neighbors[EAST]  == MPI_PROC_NULL; };

    inline constexpr bool isNotTopRow()      { return _neighbors[NORTH] != MPI_PROC_NULL; };
    inline constexpr bool isNotBottomRow()   { return _neighbors[SOUTH] != MPI_PROC_NULL; };
    inline constexpr bool isNotLeftColumn()  { return _neighbors[WEST]  != MPI_PROC_NULL; };
    inline constexpr bool isNotRightColumn() { return _neighbors[EAST]  != MPI_PROC_NULL; };
};

LifeSimulation::LifeSimulation(int argc, char **argv)
{
    parseArguments(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &_worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &_worldSize);

    constructMeshTopology();
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
        destructMeshTopology();
    }
}

void LifeSimulation::parseArguments(int argc, char **argv)
{
    // lambda function for printing an error message when a required argument is missing
    auto missingArgumentError = [&](const string &name)
    {
        if (_worldRank == ROOT)
        {
            cerr << "Error: Missing argument for the '" << name << "' switch." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    };

    // lambda function for parsing an integer argument
    auto parseInt = [&](const string &argument, const string &name) -> int
    {
        int value = 0;
        try
        {
            value = stoi(argument);
        }
        catch (const invalid_argument& e)
        {
            if (_worldRank == ROOT)
            {
                cerr << "Invalid argument for the '" << name << "' switch." << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        catch (const out_of_range& e)
        {
            if (_worldRank == ROOT)
            {
                cerr << "Argument for the '" << name << "' switch is out of range." << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (value < 0)
        {
            if (_worldRank == ROOT)
            {
                cerr << "Argument for the '" << name << "' switch must be a non-negative number." << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        return value;
    };
    
    // convert arguments to a vector of strings
    vector<string> arguments(argv, argv + argc);

    if (arguments.size() < 3)
    {
        if (_worldRank == ROOT)
        {
            cerr << "Error: Not enough arguments (missing 'grid file name' and/or 'number of iterations')." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    _arguments.inputFileName = arguments[1];
    try
    {
        _arguments.numberOfIterations = stoi(arguments[2]);
    }
    catch (const invalid_argument& e)
    {
        if (_worldRank == ROOT)
        {
            cerr << "Invalid 2nd argument (number of iterations)." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    } 
    catch (const out_of_range& e)
    {
        if (_worldRank == ROOT)
        {
            cerr << "Number of iterations is out of range." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    if (_arguments.numberOfIterations < 0)
    {
        if (_worldRank == ROOT)
        {
            cerr << "Number of iterations must be a non-negative unmber." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // parse additional arguments
    for (auto iterator = arguments.begin() + 3; iterator != arguments.end(); iterator++)
    {
        auto lastIterator = iterator;
        if (*iterator == "-w" || *iterator == "--wraparound")
        {
            _arguments.wraparound = true;
        }
        else if (*iterator == "-dx" || *iterator == "--decomposition_x")
        {
            iterator++;
            if (iterator == arguments.end())
            {
                missingArgumentError(*lastIterator);
            }

            _arguments.nodesWidthCount = parseInt(*iterator, *lastIterator);
        }
        else if (*iterator == "-dy" || *iterator == "--decomposition_y")
        {
            iterator++;
            if (iterator == arguments.end())
            {
                missingArgumentError(*lastIterator);
            }

            _arguments.nodesHeightCount = parseInt(*iterator, *lastIterator);
        }
        else if (*iterator == "-p" || *iterator == "--padding")
        {
            iterator++;
            if (iterator == arguments.end())
            {
                missingArgumentError(*lastIterator);
            }

            _arguments.padding = parseInt(*iterator, *lastIterator);

            // write-through the padding also to the specific paddings if not specified yet
            if (_arguments.paddingHeight == 0)
            {
                _arguments.paddingHeight = _arguments.padding;
            }
            if (_arguments.paddingWidth == 0)
            {
                _arguments.paddingWidth = _arguments.padding;
            }
        }
        else if (*iterator == "px" || *iterator == "--padding_x")
        {
            iterator++;
            if (iterator == arguments.end())
            {
                missingArgumentError(*lastIterator);
            }

            _arguments.paddingWidth = parseInt(*iterator, *lastIterator);
        }
        else if (*iterator == "py" || *iterator == "--padding_y")
        {
            iterator++;
            if (iterator == arguments.end())
            {
                missingArgumentError(*lastIterator);
            }

            _arguments.paddingHeight = parseInt(*iterator, *lastIterator);
        }
        else
        {
            if (_worldRank == ROOT)
            {
                cerr << "Warning: Unknown switch '" << *iterator << "'." << endl;
            }
        }
    }
}

void LifeSimulation::constructMeshTopology()
{
    int msbPosition = sizeof(int) * 8 - 1 - __builtin_clz(_worldSize); // get the number of processes which is a power of 2

    if (_arguments.nodesHeightCount == 0 && _arguments.nodesWidthCount == 0) // topology dimensions not specified
    {
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
    }
    else if (_arguments.nodesHeightCount != 0 && _arguments.nodesWidthCount != 0) // topology dimensions specified fully
    {
        if (_arguments.nodesHeightCount * _arguments.nodesWidthCount > _worldSize)
        {
            if (_worldRank == ROOT)
            {
                cerr << "Error: The number of processes specified by the topology dimensions is greater than the number of started processes." << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        _settings.nodesWidthCount = _arguments.nodesWidthCount;
        _settings.nodesHeightCount = _arguments.nodesHeightCount;
    }
    else // topology dimensions specified partially
    {
        if (_arguments.nodesHeightCount == 0)
        {
            _settings.nodesWidthCount = _arguments.nodesWidthCount;
            _settings.nodesHeightCount = _worldSize / _settings.nodesWidthCount;
        }
        else
        {
            _settings.nodesHeightCount = _arguments.nodesHeightCount;
            _settings.nodesWidthCount = _worldSize / _settings.nodesHeightCount;
        }
    }
    _settings.nodesTotalCount = _settings.nodesWidthCount * _settings.nodesHeightCount;
    
    int color = _worldRank >= _settings.nodesTotalCount; // exclude processes that will not fit in the mesh
    if (color)
    {
        cerr << "Warning: Process " << _worldRank << " is not going to be part of the mesh, i.e. will be idle." << endl;
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, MPI_UNDEFINED, &_subWorldCommunicator);
        _subWorldCommunicator = MPI_COMM_NULL;
        _meshCommunicator = MPI_COMM_NULL;

        return; // this process will no longer be utilized
    }
    else
    {
        MPI_Comm_split(MPI_COMM_WORLD, color, _worldRank, &_subWorldCommunicator);
    }

    // create a 2D mesh topology
    MPI_Cart_create(_subWorldCommunicator, 2, 
                    array<int, 2>{_settings.nodesHeightCount, _settings.nodesWidthCount}.data(), 
                    array<int, 2>{_arguments.wraparound, _arguments.wraparound}.data(), 
                    0, &_meshCommunicator);
    // retrieve the rank and size of the mesh communicator
    MPI_Comm_rank(_meshCommunicator, &_meshRank);
    MPI_Comm_size(_meshCommunicator, &_meshSize);

    // retrieve the neighbors of the current process
    MPI_Cart_shift(_meshCommunicator, 0, 1, &_neighbors[NORTH], &_neighbors[SOUTH]);
    MPI_Cart_shift(_meshCommunicator, 1, 1, &_neighbors[WEST], &_neighbors[EAST]);

    // retrieve the coordinates of the current process
    int coordinates[2];
    MPI_Cart_coords(_meshCommunicator, _meshRank, 2, coordinates);

    // left top corner (NW)
    coordinates[0]--;
    coordinates[1]--;
    if (coordinates[0] >= 0 && coordinates[1] >= 0)
    {
        MPI_Cart_rank(_meshCommunicator, coordinates, &_cornerNeighbors[LEFT_UPPER]);
    }
    else if (_arguments.wraparound)
    {
        coordinates[0] = (coordinates[0] + _settings.nodesHeightCount) % _settings.nodesHeightCount;
        coordinates[1] = (coordinates[1] + _settings.nodesWidthCount) % _settings.nodesWidthCount;
        MPI_Cart_rank(_meshCommunicator, coordinates, &_cornerNeighbors[LEFT_UPPER]);
    }
    else
    {
        _cornerNeighbors[LEFT_UPPER] = MPI_PROC_NULL;
    }

    // right top corner (NE)
    coordinates[1] += 2;
    if (coordinates[0] >= 0 && coordinates[1] < _settings.nodesWidthCount)
    {
        MPI_Cart_rank(_meshCommunicator, coordinates, &_cornerNeighbors[RIGHT_UPPER]);
    }
    else if (_arguments.wraparound)
    {
        coordinates[1] = coordinates[1] - _settings.nodesWidthCount;
        MPI_Cart_rank(_meshCommunicator, coordinates, &_cornerNeighbors[RIGHT_UPPER]);
    }
    else
    {
        _cornerNeighbors[RIGHT_UPPER] = MPI_PROC_NULL;
    }

    // right bottom corner (SE)
    coordinates[0] += 2;
    if (coordinates[0] < _settings.nodesHeightCount && coordinates[1] < _settings.nodesWidthCount)
    {
        MPI_Cart_rank(_meshCommunicator, coordinates, &_cornerNeighbors[RIGHT_LOWER]);
    }
    else if (_arguments.wraparound)
    {
        coordinates[0] = coordinates[0] - _settings.nodesHeightCount;
        MPI_Cart_rank(_meshCommunicator, coordinates, &_cornerNeighbors[RIGHT_LOWER]);
    }
    else
    {
        _cornerNeighbors[RIGHT_LOWER] = MPI_PROC_NULL;
    }

    // left bottom corner (SW)
    coordinates[1] -= 2;
    if (coordinates[0] < _settings.nodesHeightCount && coordinates[1] >= 0)
    {
        MPI_Cart_rank(_meshCommunicator, coordinates, &_cornerNeighbors[LEFT_LOWER]);
    }
    else if (_arguments.wraparound)
    {
        coordinates[1] = coordinates[1] + _settings.nodesWidthCount;
        MPI_Cart_rank(_meshCommunicator, coordinates, &_cornerNeighbors[LEFT_LOWER]);
    }
    else
    {
        _cornerNeighbors[LEFT_LOWER] = MPI_PROC_NULL;
    }
}

void LifeSimulation::destructMeshTopology()
{
    if (_subWorldCommunicator != MPI_COMM_NULL)
    {
        MPI_Comm_free(&_subWorldCommunicator);
    }

    if (_meshCommunicator != MPI_COMM_NULL)
    {
        MPI_Comm_free(&_meshCommunicator);
    }
}

void LifeSimulation::readInputFile()
{
    if (_worldRank == ROOT) // only the root process reads the input file
    {
        ifstream inputFile(_arguments.inputFileName, ios::in);
        if (!inputFile.is_open())
        {
            cerr << "Error: Unable to open input file '" << _arguments.inputFileName << "'." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // get the size of the input file
        inputFile.seekg(0, ios::end);
        int fileSize = inputFile.tellg();
        inputFile.seekg(0, ios::beg);

        string row;
        getline(inputFile, row);

        // determine the dimensions of the grid based on the first row
        _settings.globalNotPaddedWidth = row.length();
        _settings.globalNotPaddedHeight = (fileSize + 1) / (_settings.globalNotPaddedWidth + 1); // +1 for newline characters
        if (_settings.globalNotPaddedHeight * (_settings.globalNotPaddedWidth + 1) != fileSize && 
            _settings.globalNotPaddedHeight * (_settings.globalNotPaddedWidth + 1) - 1 != fileSize)
        {
            cerr << "Error: The input file must contain a rectangular grid of '1' and '0' characters." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // adjust the dimensions of the grid to be divisible by the number of nodes in the mesh, also add additional padding if specified
        _settings.globalWidth = ((_settings.globalNotPaddedWidth + _settings.nodesWidthCount - 1 + 2 * _arguments.paddingWidth) / 
                                 _settings.nodesWidthCount) * _settings.nodesWidthCount;
        _settings.globalHeight = ((_settings.globalNotPaddedHeight + _settings.nodesHeightCount - 1  + 2 * _arguments.paddingHeight) / 
                                  _settings.nodesHeightCount) * _settings.nodesHeightCount;
        
        if (_settings.globalWidth != _settings.globalNotPaddedWidth + 2 * _arguments.padding)
        {
            cerr << "Warning: The input file X dimension (width) of " << _settings.globalNotPaddedWidth 
                 << (_arguments.paddingWidth ? " with the additional padding " : " ") 
                 << "is not divisible by the X decomposition dimension." << endl;
            cerr << "         The grid X dimension will be extended to " << _settings.globalWidth << "." << endl;
        }
        if (_settings.globalHeight != _settings.globalNotPaddedHeight + 2 * _arguments.padding)
        {
            cerr << "Warning: The input file Y dimension (height) of " << _settings.globalNotPaddedHeight 
                 << (_arguments.paddingHeight ? " with the additional padding " : " ") << "is not divisible by the Y decomposition dimension." << endl;
            cerr << "         The grid Y dimension will be extended to " << _settings.globalHeight << "." << endl;
        }
        
        _settings.localWidth = _settings.globalWidth / _settings.nodesWidthCount;
        _settings.localHeight = _settings.globalHeight / _settings.nodesHeightCount;
        _settings.localWidthWithHaloZones = _settings.localWidth + 2;
        _settings.localHeightWithHaloZones = _settings.localHeight + 2;
        _settings.localTileSize = _settings.localHeight * _settings.localWidth;
        _settings.localTileSizeWithHaloZones = _settings.localHeightWithHaloZones * _settings.localWidthWithHaloZones;

        // offsets in the global grid of the read data
        int rowPadding = (_settings.globalWidth - _settings.globalNotPaddedWidth) >> 1;
        int colPadding = (_settings.globalHeight - _settings.globalNotPaddedHeight) >> 1;

        #ifdef DEBUG_PRINT
            cerr << "Number of iterations:     " << _arguments.numberOfIterations << endl;
            cerr << "globalHeight:             " << _settings.globalHeight << endl;
            cerr << "globalWidth:              " << _settings.globalWidth << endl;
            cerr << "globalNotPaddedHeight:    " << _settings.globalNotPaddedHeight << endl;
            cerr << "globalNotPaddedWidth:     " << _settings.globalNotPaddedWidth << endl;
            cerr << "localHeight:              " << _settings.localHeight << endl;
            cerr << "localWidth:               " << _settings.localWidth << endl;
            cerr << "localHeightWithHaloZones: " << _settings.localHeightWithHaloZones << endl;
            cerr << "localWidthWithHaloZones:  " << _settings.localWidthWithHaloZones << endl;
            cerr << "nodesHeightCount:         " << _settings.nodesHeightCount << endl;
            cerr << "nodesWidthCount:          " << _settings.nodesWidthCount << endl;
            cerr << "nodesTotalCount:          " << _settings.nodesTotalCount << endl;
        #endif

        // allocate and initialize the global tile
        _globalTile = static_cast<cell_t *>(aligned_alloc(64, _settings.globalHeight * _settings.globalWidth * sizeof(cell_t)));
        if (_globalTile == nullptr)
        {
            cerr << "Error: Unable to allocate memory for the global tile." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        memset(_globalTile, 0, _settings.globalHeight * _settings.globalWidth * sizeof(cell_t));

        int readLines = 0;
        int idx = colPadding * _settings.globalWidth;
        do
        {
            idx += rowPadding;
            if (row.length() != static_cast<size_t>(_settings.globalNotPaddedWidth)) // unexpected row length
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
    int outerTileSizes[2] = {_settings.globalHeight, _settings.globalWidth};
    int innerTileSizes[2] = {_settings.localHeight, _settings.localWidth};
    int starts[2] = {0, 0};
    
    // tile data type for scatter and gather data type
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_BYTE, &_tileType);
    MPI_Type_commit(&_tileType);
    MPI_Type_create_resized(_tileType, 0, _settings.localWidth * sizeof(cell_t), &_tileResizedType);
    MPI_Type_commit(&_tileResizedType);

    // tile data type with halo zones
    outerTileSizes[0] = _settings.localHeightWithHaloZones;
    outerTileSizes[1] = _settings.localWidthWithHaloZones;
    starts[0] = 1;
    starts[1] = 1;
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_BYTE, &_tileWithHaloZonesType);
    MPI_Type_commit(&_tileWithHaloZonesType);
    
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

    if (_arguments.wraparound && _settings.nodesHeightCount < 3) // wraparound is enabled and there are less than two rows of nodes
    {
        // this is a weird behaviour of the MPI wraparound, where for less than 3 rows the messages are send in reverse order
        swap(_sendHaloZoneTypes[SOUTH], _sendHaloZoneTypes[NORTH]);
    }

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

    if (_arguments.wraparound && _settings.nodesWidthCount < 3) // wraparound is enabled and there are less than two columns of nodes
    {
        // this is a weird behaviour of the MPI wraparound, where for less than 3 columns the messages are send in reverse order
        swap(_sendHaloZoneTypes[EAST], _sendHaloZoneTypes[WEST]);
    }
}

void LifeSimulation::destructDataTypes()
{
    MPI_Type_free(&_tileType);
    MPI_Type_free(&_tileResizedType);
    MPI_Type_free(&_tileWithHaloZonesType);

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
    _currentTile = static_cast<cell_t *>(aligned_alloc(64, _settings.localTileSizeWithHaloZones * sizeof(cell_t)));
    _nextTile    = static_cast<cell_t *>(aligned_alloc(64, _settings.localTileSizeWithHaloZones * sizeof(cell_t)));
    if (_currentTile == nullptr || _nextTile == nullptr)
    {
        cerr << "Error: Unable to allocate memory for local tiles." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ensure the next tile is initialized, data will be copied from the next tile into the second tile later after initial halo zones exchange
    memset(_nextTile, 0, _settings.localTileSizeWithHaloZones * sizeof(cell_t));
}

void LifeSimulation::destructGridTiles()
{
    if (_currentTile != nullptr)
    {
        free(_currentTile);
    }

    if (_nextTile != nullptr)
    {
        free(_nextTile);
    }
}

void LifeSimulation::exchangeInitialData()
{
    // partition the global tile into local tiles and scatter them to all processes in the mesh
    MPI_Scatterv(_globalTile, _scatterGatherCounts.data(), _scatterGatherDisplacements.data(), _tileResizedType, 
                 _nextTile, 1, _tileWithHaloZonesType, ROOT, _meshCommunicator);
    // perform the initial exchange of halo zones
    startHaloZonesExchange();
    // wait for the initial exchange of halo zones to finish
    awaitHaloZonesExchange();
    // copy the initial data into the second tile
    copy(_nextTile, _nextTile + _settings.localTileSizeWithHaloZones, _currentTile);
}

void LifeSimulation::startHaloZonesExchange()
{
    // initiate corner exchanges
    if (_cornerNeighbors[LEFT_UPPER] != MPI_PROC_NULL)
    {
        MPI_Isend(_nextTile + _settings.localWidthWithHaloZones + 1, 1, MPI_BYTE, _cornerNeighbors[LEFT_UPPER], LEFT_UPPER, 
                  _meshCommunicator, &_cornerSendRequests[LEFT_UPPER]);
        MPI_Irecv(_nextTile, 1, MPI_BYTE, _cornerNeighbors[LEFT_UPPER], RIGHT_LOWER, 
                  _meshCommunicator, &_cornerRecvRequests[LEFT_UPPER]);
    }
    else
    {
        _cornerSendRequests[LEFT_UPPER] = MPI_REQUEST_NULL;
        _cornerRecvRequests[LEFT_UPPER] = MPI_REQUEST_NULL;
    }

    if (_cornerNeighbors[RIGHT_UPPER] != MPI_PROC_NULL)
    {
        MPI_Isend(_nextTile +  2 * _settings.localWidthWithHaloZones - 2, 1, MPI_BYTE, _cornerNeighbors[RIGHT_UPPER], RIGHT_UPPER, 
                  _meshCommunicator, &_cornerSendRequests[RIGHT_UPPER]);
        MPI_Irecv(_nextTile + _settings.localWidthWithHaloZones - 1, 1, MPI_BYTE, _cornerNeighbors[RIGHT_UPPER], LEFT_LOWER, 
                  _meshCommunicator, &_cornerRecvRequests[RIGHT_UPPER]);
    }
    else
    {
        _cornerSendRequests[RIGHT_UPPER] = MPI_REQUEST_NULL;
        _cornerRecvRequests[RIGHT_UPPER] = MPI_REQUEST_NULL;
    }

    if (_cornerNeighbors[RIGHT_LOWER] != MPI_PROC_NULL)
    {
        MPI_Isend(_nextTile + _settings.localTileSizeWithHaloZones - _settings.localWidthWithHaloZones - 2, 1, MPI_BYTE, _cornerNeighbors[RIGHT_LOWER], RIGHT_LOWER, 
                  _meshCommunicator, &_cornerSendRequests[RIGHT_LOWER]);
        MPI_Irecv(_nextTile + _settings.localTileSizeWithHaloZones - 1, 1, MPI_BYTE, _cornerNeighbors[RIGHT_LOWER], LEFT_UPPER,
                  _meshCommunicator, &_cornerRecvRequests[RIGHT_LOWER]);
    }
    else
    {
        _cornerSendRequests[RIGHT_LOWER] = MPI_REQUEST_NULL;
        _cornerRecvRequests[RIGHT_LOWER] = MPI_REQUEST_NULL;
    }

    if (_cornerNeighbors[LEFT_LOWER] != MPI_PROC_NULL)
    {
        MPI_Isend(_nextTile + _settings.localTileSizeWithHaloZones - 2 * _settings.localWidthWithHaloZones + 1, 1, MPI_BYTE, _cornerNeighbors[LEFT_LOWER], LEFT_LOWER, 
                  _meshCommunicator, &_cornerSendRequests[LEFT_LOWER]);
        MPI_Irecv(_nextTile + _settings.localTileSizeWithHaloZones - _settings.localWidthWithHaloZones, 1, MPI_BYTE, _cornerNeighbors[LEFT_LOWER], RIGHT_UPPER,
                  _meshCommunicator, &_cornerRecvRequests[LEFT_LOWER]);
    }
    else
    {
        _cornerSendRequests[LEFT_LOWER] = MPI_REQUEST_NULL;
        _cornerRecvRequests[LEFT_LOWER] = MPI_REQUEST_NULL;
    }

    // initiate halo zone exchanges
    MPI_Ineighbor_alltoallw(_nextTile, _neighbourCounts, _neighbourDisplacements, _sendHaloZoneTypes, _nextTile, 
                            _neighbourCounts, _neighbourDisplacements, _recvHaloZoneTypes, _meshCommunicator, &_haloZoneRequest);
}

void LifeSimulation::awaitHaloZonesExchange()
{
    // await corner exchanges
    MPI_Waitall(4, _cornerSendRequests, MPI_STATUSES_IGNORE);
    MPI_Waitall(4, _cornerRecvRequests, MPI_STATUSES_IGNORE);

    // await halo zone exchanges
    MPI_Wait(&_haloZoneRequest, MPI_STATUS_IGNORE);
}

void LifeSimulation::computeHaloZones()
{
    // unpack the source tile for easier access
    cell_t *currentTile = _currentTile;
    cell_t *currentTopRow0 = currentTile; // closest to the top
    cell_t *currentTopRow1 = currentTile + _settings.localWidthWithHaloZones;
    cell_t *currentTopRow2 = currentTile + 2 * _settings.localWidthWithHaloZones;
    cell_t *currentBottomRow0 = currentTile + _settings.localTileSizeWithHaloZones - 3 * _settings.localWidthWithHaloZones;
    cell_t *currentBottomRow1 = currentTile + _settings.localTileSizeWithHaloZones - 2 * _settings.localWidthWithHaloZones;
    cell_t *currentBottomRow2 = currentTile + _settings.localTileSizeWithHaloZones - _settings.localWidthWithHaloZones; // closest to the bottom

    // unpack the destination tile for easier access
    cell_t *nextTile = _nextTile;
    cell_t *nextTopRow1 = nextTile + _settings.localWidthWithHaloZones;
    cell_t *nextBottomRow1 = nextTile + _settings.localTileSizeWithHaloZones - 2 * _settings.localWidthWithHaloZones;

    // north halo zone
    for (int i = 1; i < _settings.localWidthWithHaloZones - 1; i++) // adjust offset to not compute undefined corners
    {
        nextTopRow1[i] = updateCell(currentTopRow0[i - 1], currentTopRow0[i], currentTopRow0[i + 1], 
                                    currentTopRow1[i - 1], currentTopRow1[i], currentTopRow1[i + 1], 
                                    currentTopRow2[i - 1], currentTopRow2[i], currentTopRow2[i + 1]);
    }

    // west halo zone
    for (int i = 2; i < _settings.localHeightWithHaloZones - 2; i++) // adjust offset to not compute already computed values
    {
        int topIdx = (i - 1) * _settings.localWidthWithHaloZones;
        int centerIdx = i * _settings.localWidthWithHaloZones;
        int bottomIdx = (i + 1) * _settings.localWidthWithHaloZones;
        nextTile[centerIdx + 1] = updateCell(currentTile[topIdx], currentTile[topIdx + 1], currentTile[topIdx + 2],
                                             currentTile[centerIdx], currentTile[centerIdx + 1], currentTile[centerIdx + 2],
                                             currentTile[bottomIdx], currentTile[bottomIdx + 1], currentTile[bottomIdx + 2]);
    }

    // east halo zone
    for (int i = 3; i < _settings.localHeightWithHaloZones - 1; i++) // adjust offset to not compute already computed values
    {
        int topIdx = (i - 1) * _settings.localWidthWithHaloZones - 3;
        int centerIdx = i * _settings.localWidthWithHaloZones - 3;
        int bottomIdx = (i + 1) * _settings.localWidthWithHaloZones - 3;
        nextTile[centerIdx + 1] = updateCell(currentTile[topIdx], currentTile[topIdx + 1], currentTile[topIdx + 2],
                                             currentTile[centerIdx], currentTile[centerIdx + 1], currentTile[centerIdx + 2],
                                             currentTile[bottomIdx], currentTile[bottomIdx + 1], currentTile[bottomIdx + 2]);
    }
    
    // south halo zone
    for (int i = 1; i < _settings.localWidthWithHaloZones - 1; i++)
    {
        nextBottomRow1[i] = updateCell(currentBottomRow0[i - 1], currentBottomRow0[i], currentBottomRow0[i + 1], 
                                       currentBottomRow1[i - 1], currentBottomRow1[i], currentBottomRow1[i + 1], 
                                       currentBottomRow2[i - 1], currentBottomRow2[i], currentBottomRow2[i + 1]);
    }
}

void LifeSimulation::computeTile()
{
    cell_t *currentTile = _currentTile;
    cell_t *nextTile = _nextTile;

    for (int i = 2; i < _settings.localHeightWithHaloZones - 2; i++)
    {
        int topIdx = (i - 1) * _settings.localWidthWithHaloZones;
        int centerIdx = i * _settings.localWidthWithHaloZones;
        int bottomIdx = (i + 1) * _settings.localWidthWithHaloZones;

        #pragma omp simd aligned(currentTile, nextTile: 64) simdlen(16)
        for (int j = 2; j < _settings.localWidthWithHaloZones - 2; j++)
        {
            nextTile[centerIdx + j] = updateCell(currentTile[topIdx + j - 1], currentTile[topIdx + j], currentTile[topIdx + j + 1],
                                                 currentTile[centerIdx + j - 1], currentTile[centerIdx + j], currentTile[centerIdx + j + 1],
                                                 currentTile[bottomIdx + j - 1], currentTile[bottomIdx + j], currentTile[bottomIdx + j + 1]);
        }
    }
}

void LifeSimulation::collectResults()
{
    // gather the local tiles into the global tile
    MPI_Gatherv(_currentTile, 1, _tileWithHaloZonesType, _globalTile, _scatterGatherCounts.data(), _scatterGatherDisplacements.data(), 
                _tileResizedType, ROOT, _meshCommunicator);
}

void LifeSimulation::debugPrintLocalTile(bool current = true)
{
    cell_t *tile = current ? _currentTile : _nextTile;
    for (int node = 0; node < _meshSize; node++)
    {
        if (_meshRank == node)
        {
            cerr << "Rank: " << _meshRank << "\n";
            for (int i = 0; i < _settings.localHeightWithHaloZones; i++)
            {
                for (int j = 0; j < _settings.localWidthWithHaloZones; j++)
                {
                    if (j == 1 || j == _settings.localWidthWithHaloZones - 1) // separate west and east halo zones
                    {
                        cerr << " ";
                    }
                    cerr << static_cast<int>(tile[i * _settings.localWidthWithHaloZones + j]);
                }
                cerr << "\n";
                
                if (i == 0 || i == _settings.localHeightWithHaloZones - 2) // separate north and south halo zones
                {
                    cerr << "\n";
                }
            }
            cerr << endl;
        }
        MPI_Barrier(_meshCommunicator);
    }
}

void LifeSimulation::testPrintGlobalTile()
{
    if (_worldRank == ROOT)
    {
        for (int i = 0; i < _settings.globalHeight; i++)
        {
            for (int j = 0; j < _settings.globalWidth; j++)
            {
                cerr << static_cast<int>(_globalTile[i * _settings.globalWidth + j]);
            }
            cerr << "\n";
        }
    }
}

void LifeSimulation::prettyPrintGlobalTile()
{
    // format table header string with a process rank
    auto fillInProcessRank = [](int processId, string &target)
    {
        string processIdStr = to_string(processId);

        // find the middle of the target string
        size_t processIdLength = processIdStr.length();
        size_t halfTargetLength = (target.length() + 1) >> 1;
        size_t startIdx = halfTargetLength - ((processIdLength + 1) >> 1);

        // fill in the process id to the target string
        for (size_t i = 0; i < processIdLength && i < target.length(); i++)
        {
            target[startIdx + i] = processIdStr[i];
        }
    };

    if (_worldRank == ROOT)
    {
        string horizontalSeparator(_settings.globalWidth + _settings.nodesWidthCount + 2, '-');
        string horizontalProcessId(_settings.localWidth, ' ');
        string verticalProcessId(_settings.localHeight, ' ');

        cout << " ";
        for (int i = 0; i < _settings.nodesWidthCount; i++)
        {
            cout << "|";
            fillInProcessRank(i, horizontalProcessId);
            cout << horizontalProcessId;
        }
        cout << "|\n";


        for (int i = 0; i < _settings.nodesHeightCount; i++)
        {
            cout << horizontalSeparator << "\n";
            fillInProcessRank(i, verticalProcessId);

            for (int j = 0; j < _settings.localHeight; j++)
            {
                cout << verticalProcessId[j] << "|";
                for (int k = 0; k < _settings.nodesWidthCount; k++)
                {
                    for (int l = 0; l < _settings.localWidth; l++)
                    {
                        cout << static_cast<int>(_globalTile[(i * _settings.localHeight + j) * _settings.globalWidth + k * _settings.localWidth + l]);
                    }
                    cout << "|";
                }
                cout << "\n";
            }
        }

        cout << horizontalSeparator << endl;
    }
}

void LifeSimulation::run()
{
    if (!isActiveProcess()) // idle processes do not participate in the simulation
    {
        return;
    }

    exchangeInitialData();

    // simulation loop
    for (int iteration = 0; iteration < _arguments.numberOfIterations; iteration++)
    {   
        computeHaloZones();
        startHaloZonesExchange();
        computeTile();
        awaitHaloZonesExchange();

        swap(_currentTile, _nextTile);
    }
    
    collectResults();
    #ifdef TEST_PRINT
        testPrintGlobalTile();
    #endif
    prettyPrintGlobalTile();
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    {
        LifeSimulation simulation(argc, argv);
        simulation.run();
    } // ensure all resources are deallocated before MPI_Finalize

    MPI_Finalize();
    return 0;
}
