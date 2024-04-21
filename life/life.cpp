// =======================================================================================================================================================
// Project:         John Conway's Game of Life simulation
// Author:          David Mihola (xmihol00)
// E-mail:          xmihol00@stud.fit.vutbr.cz
// Date:            26. 4. 2024
// Description:     A parallel implementation of the John Conway's Game of Life using the MPI library for communication. 
// N° of processes: Arbitrary, HOWEVER THE PLAYING GRID MAY BE EXTENDED, see the description of the solution below.
// Repository:      https://github.com/xmihol00/MPI_projects/tree/main/life
// Makefile:        https://github.com/xmihol00/MPI_projects/blob/main/life/Makefile
// Tests:           https://github.com/xmihol00/MPI_projects/blob/main/life/test_against_solved.sh
//                  https://github.com/xmihol00/MPI_projects/blob/main/life/wraparound_test_against_solved.sh
// =======================================================================================================================================================

// =======================================================================================================================================================
// Usage: mpiexec -np <number of processes> ./life <grid file name> <number of iterations> [options]
// Options:
//   -w,   --wraparound                Use wrapped around simulation.
//   -nx,  --nodes_x <number>          Number of nodes (processes) in the X direction of the mesh topology.
//   -ny,  --nodes_y <number>          Number of nodes (processes) in the Y direction of the mesh topology.
//   -p,   --padding <number>          Padding of the global grid in all directions.
//   -px,  --padding_x <number>        Padding of the global grid in the X (height) direction.
//   -py,  --padding_y <number>        Padding of the global grid in the Y (width) direction.
//   -pt,  --padding_top <number>      Padding of the global grid from the top.
//   -pb,  --padding_bottom <number>   Padding of the global grid from the bottom.
//   -pl,  --padding_left <number>     Padding of the global grid from the left.
//   -pr,  --padding_right <number>    Padding of the global grid from the right.
//   -ppc, --pixels_per_cell <number>  Number of pixels per cell in the generated images/video.
//   -iod, --images_output_directory <directory>
//                                     Directory where generated images will be stored to.
//   -v,   --video <file name>         Generate a video of the simulation in mp4 format.
//   -fps, --frames_per_second <number>
//                                     Frames per second in the generated video.
//   -nfp, --no_formatted_print        Do not print the global grid in a table-like format.
//   -ep,  --stderr_print              Print the unformatted global grid to stderr.
//   -h,   --help                      Print this help message.
// 
// Examples:
//   mpiexec -n 4 ./life other_grids/glider_create_gun.txt 600 -v glider_gun.mp4 -fps 10 -pb 9 -pr 3
//     # produced video: https://github.com/xmihol00/MPI_projects/blob/main/life/glider_gun.mp4
//   mpiexec --oversubscribe -n 6 ./life wraparound_solved_grids/glider_8x8/00.txt 900 -w -v glider.mp4 -p 10 -fps 15 -ny 3
//     # produced video: https://github.com/xmihol00/MPI_projects/blob/main/life/glider.mp4
//   mpiexec --oversubscribe -n 64 ./life wraparound_solved_grids/glider_8x8/00.txt 11 -w 
//     # extreme case with number of cells equal to the number of processes, output:
//      |0|1|2|3|4|5|6|7|
//     ------------------
//     0|1|0|0|0|0|0|0|0|
//     ------------------
//     1|0|1|1|0|0|0|0|0|
//     ------------------
//     2|1|1|0|0|0|0|0|0|
//     ------------------
//     3|0|0|0|0|0|0|0|0|
//     ------------------
//     4|0|0|0|0|0|0|0|0|
//     ------------------
//     5|0|0|0|0|0|0|0|0|
//     ------------------
//     6|0|0|0|0|0|0|0|0|
//     ------------------
//     7|0|0|0|0|0|0|0|0|
//     ------------------
// =======================================================================================================================================================

// =======================================================================================================================================================
// Description of the solution:
// 1. Processes are organized into a 2D (cartesian) mesh topology. The dimensions of the mesh can be specified by the user, or are derived automatically. 
//    The automatically derived mesh dimensions will always contain a power of 2 processes. Processes that do not fit into the mesh will not be utilized. 
//    Lastly, each process in the mesh retrieves ranks of its neighbors, especially of its corner neighbors (NW, NE, SE, SW).
// 2. The input file is read only by the root process. Based on the size of the input file and the length of the first row, the global grid dimensions, 
//    i.e. the space of the simulation, are determined. The global grid dimensions are adjusted to be divisible by the number of nodes in each dimension 
//    of the mesh. The content of the file is then placed to the left upper corner of the global grid. The rest is padded with '0'.
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
#include <bits/stdc++.h>
#include <fstream>

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
    /**
     * @brief Constructs the LifeSimulation object based on the user specified arguments.
     * @param argc Number of arguments (after being stripped by MPI_Initialize).
     * @param argv Array of arguments (after being stripped by MPI_Initialize).
     */
    LifeSimulation(int argc, char **argv);

    /**
     * @brief Destructs the LifeSimulation object. Must be called before the MPI_Finalize, i.e use RAII correctly.
     */
    ~LifeSimulation();

    /**
     * @brief Runs the John Conway's Game of Life simulation.
     */
    void run();

private:
    /**
     * @brief Parses the command line arguments.
     * @param argc Number of arguments (after being stripped by MPI_Initialize).
     * @param argv Array of arguments (after being stripped by MPI_Initialize).
     */
    void parseArguments(int argc, char **argv);

    /**
     * @brief Constructs a 2D mesh topology. The dimensions can be user specified via arguments or derived automatically from the number of processes.
     */
    void constructMeshTopology();

    /**
     * @brief Destructs the 2D mesh topology.
     */
    void destructMeshTopology();

    /**
     * @brief Reads the input file and constructs the global grid.
     */
    void readInputFile();

    /**
     * @brief Constructs MPI data types for the local tile, tile with halo zones, and halo zones.
     */
    void constructDataTypes();

    /**
     * @brief Destructs MPI data types.
     */
    void destructDataTypes();

    /**
     * @brief Allocates local tiles, private to each process.
     */
    void constructGridTiles();

    /**
     * @brief Deallocates local tiles.
     */
    void destructGridTiles();

    /**
     * @brief Exchanges the initial data, i.e. scatters the global grid to the local tiles of all processes and performs the initial halo zone exchange.
     */
    void exchangeInitialData();

    /**
     * @brief Starts the non-blocking exchange of halo zones.
     */
    void startHaloZonesExchange();

    /**
     * @brief Awaits the completion of the non-blocking exchange of halo zones.
     */
    void awaitHaloZonesExchange();

    /**
     * @brief Computes the next state of the simulation in the edges of the local tile.
     */
    void computeHaloZones();

    /**
     * @brief Computes the next state of the simulation in the inner part of the local tile.
     */
    void computeTile();

    /**
     * @brief Gathers the current state of the simulation from all processes to the root process.
     */
    void collectLocalTiles();

    /**
     * @brief Updates the state of a cell based on the states of its neighbors.
     */
    inline constexpr cell_t updateCell
    (
        cell_t northWest, cell_t north,  cell_t northEast, 
        cell_t west,      cell_t center, cell_t east, 
        cell_t southWest, cell_t south,  cell_t southEast
    );

    /**
     * @brief Prints local tiles in the order of ranks, separating each rank with a barrier call.
     */
    void debugPrintLocalTile(bool current);

    /**
     * @brief Prints the unformatted global grid to stderr, living cells represented by '1', dead cells by '0', i.e. the global grid buffer is dumped in ASCII.
     */
    void unformattedPrintGlobalTile(ostream &stream);

    /**
     * @brief Prints the global grid in a table-like format, where the table header contains coordinates of the utilized processes in the 2D mesh.
     */
    void formattedPrintGlobalTile();

    /**
     * @brief Generates an image of the current state of the simulation the PBM format.
     */
    void generateImagePBM();

    /**
     * @brief Prints a FFMPEG command to stderr for generating a video from generated images.
     */
    void printFFMPEGCommand();

    /**
     * @brief Generates a video of the simulation using FFMPEG.
     */
    void generateVideoFFMPEG();

    enum { NORTH = 0, SOUTH, WEST, EAST };                         ///< Directions in the 2D mesh.
    enum { LEFT_UPPER = 0, RIGHT_UPPER, RIGHT_LOWER, LEFT_LOWER }; ///< Corner neighbors of a process in the 2D mesh.
    static constexpr int ROOT = 0;                                 ///< Rank of the root process.
    static constexpr int NEIGHBOR_COUNT = 4;                       ///< Number of neighbors of a process in the 2D mesh.

    MPI_Comm _subWorldCommunicator = MPI_COMM_NULL; ///< Communicator containing only the processes that will take part in the simulation.
    MPI_Comm _meshCommunicator = MPI_COMM_NULL;     ///< Communicator containing the processes organized in a 2D mesh topology.

    bool _activeProcess = true; ///< Flag indicating whether the current process is going to be part of the simulation.
    int _worldRank;             ///< Rank of the current process in MPI_COMM_WORLD.
    int _worldSize;             ///< Number of processes in MPI_COMM_WORLD.
    int _meshSize;              ///< Number of processes in the mesh communicator.
    int _meshRank;              ///< Rank of the current process in the mesh communicator.

    /**
     * @brief Structure containing the arguments of the simulation specified by user.
     */
    struct Arguments
    {
        string inputFileName;             ///< Name of the input file containing the initial state of the simulation, must be user specified.
        int numberOfIterations;           ///< Number of iterations of the simulation, must be user specified.
        int padding = 0;                  ///< Padding of the global grid in all directions.
        int paddingHeight = 0;            ///< Padding of the global grid in the Y direction.
        int paddingWidth = 0;             ///< Padding of the global grid in the X direction.
        int paddingTop = 0;               ///< Padding of the global grid from the top.
        int paddingBottom = 0;            ///< Padding of the global grid from the bottom.
        int paddingLeft = 0;              ///< Padding of the global grid from the left.
        int paddingRight = 0;             ///< Padding of the global grid from the right;
        bool wraparound = false;          ///< Use wrapped around simulation.
        int nodesHeightCount = 0;         ///< Number of nodes (processes) in the Y direction of the mesh topology.
        int nodesWidthCount = 0;          ///< Number of nodes (processes) in the X direction of the mesh topology.
        int pixelsPerCell = 10;           ///< Number of pixels per cell in the generated images/video.
        int fps = 4;                      ///< Frames per second in the generated video.
        string outputImageDirectoryName;  ///< Directory where generated images will be stored to.
        string videoFileName;             ///< Name of the generated video file.
        bool formattedPrint = true;       ///< Print the global grid in a table-like format.
        #ifdef _TEST_PRINT_
            bool stderrPrint = true;      
        #else
            bool stderrPrint = false;     ///< Print the unformatted global grid to stderr.
        #endif
    } _arguments;

    /**
     * @brief Structure containing the settings of the simulation.
     */
    struct Settings
    {
        int globalHeight;                   ///< Height of the global grid.
        int globalWidth;                    ///< Width of the global grid.
        int globalNotPaddedHeight;          ///< Height of the global grid without padding.
        int globalNotPaddedWidth;           ///< Width of the global grid without padding.
        int localHeight;                    ///< Height of the local tile without halo zones.
        int localWidth;                     ///< Width of the local tile without halo zones.
        int localHeightWithHaloZones;       ///< Height of the local tile with halo zones.
        int localWidthWithHaloZones;        ///< Width of the local tile with halo zones.
        int localTileSize;                  ///< Size of the local tile without halo zones.
        int localTileSizeWithHaloZones;     ///< Size of the local tile with halo zones.
        int nodesHeightCount;               ///< Number of nodes (processes) in the Y direction of the mesh topology.
        int nodesWidthCount;                ///< Number of nodes (processes) in the X direction of the mesh topology.
        int nodesTotalCount;                ///< Total number of nodes (processes) in the mesh topology.
        bool generateImages = false;        ///< Generate images of each step of the simulation.
        bool generateOnlyVideo = false;     ///< Generate only a video of the simulation.
        bool generateVideo = false;         ///< Generate a video of the simulation.
    } _settings;

    MPI_Datatype _tileType;              ///< MPI data type for the local tile.
    MPI_Datatype _tileResizedType;       ///< MPI data type for the local tile resized allowing scatter and gather.
    MPI_Datatype _tileWithHaloZonesType; ///< MPI data type for the local tile with halo zones.

    MPI_Datatype _sendHaloZoneTypes[NEIGHBOR_COUNT]; ///< MPI data types for sending halo zones.
    MPI_Datatype _recvHaloZoneTypes[NEIGHBOR_COUNT]; ///< MPI data types for receiving halo zones.

    vector<int> _scatterGatherCounts;        ///< Counts of transferred tiles to each process during scatter and gather operations.
    vector<int> _scatterGatherDisplacements; ///< Displacements of transferred tiles to each process during scatter and gather operations.

    MPI_Request _haloZoneRequest;                    ///< MPI request for the non-blocking exchange of halo zones.
    MPI_Request _cornerSendRequests[NEIGHBOR_COUNT]; ///< MPI requests for the non-blocking exchange of corner halo zones.
    MPI_Request _cornerRecvRequests[NEIGHBOR_COUNT]; ///< MPI requests for the non-blocking exchange of corner halo zones.

    int _neighbors[NEIGHBOR_COUNT];                                  ///< Neighbors of the current process in the 2D mesh.
    int _cornerNeighbors[NEIGHBOR_COUNT];                            ///< Corner neighbors of the current process in the 2D mesh.
    int _neighbourCounts[NEIGHBOR_COUNT] = {1, 1, 1, 1};             ///< Counts of corner values to be send to each corner neighbor.
    MPI_Aint _neighbourDisplacements[NEIGHBOR_COUNT] = {0, 0, 0, 0}; ///< Displacements of corner values to be send to each corner neighbor.

    cell_t *_globalTile = nullptr;  ///< Global grid containing the whole simulation space (only the root rank allocated this buffer).
    // two tiles for ping-pong buffering and to overlap computation with communication
    cell_t *_currentTile = nullptr; ///< Local tile containing the current state of the simulation.
    cell_t *_nextTile = nullptr;    ///< Local tile containing the next state of the simulation.

    int _imageCounter = 0; ///< Counter of generated images.

    // self-explanatory helper functions
    inline constexpr bool isTopRow()      { return _neighbors[NORTH] == MPI_PROC_NULL; };
    inline constexpr bool isBottomRow()   { return _neighbors[SOUTH] == MPI_PROC_NULL; };
    inline constexpr bool isLeftColumn()  { return _neighbors[WEST]  == MPI_PROC_NULL; };
    inline constexpr bool isRightColumn() { return _neighbors[EAST]  == MPI_PROC_NULL; };

    inline constexpr bool isNotTopRow()      { return _neighbors[NORTH] != MPI_PROC_NULL; };
    inline constexpr bool isNotBottomRow()   { return _neighbors[SOUTH] != MPI_PROC_NULL; };
    inline constexpr bool isNotLeftColumn()  { return _neighbors[WEST]  != MPI_PROC_NULL; };
    inline constexpr bool isNotRightColumn() { return _neighbors[EAST]  != MPI_PROC_NULL; };
};

#pragma omp declare simd
inline constexpr cell_t LifeSimulation::updateCell
(
    cell_t northWest, cell_t north,  cell_t northEast, 
    cell_t west,      cell_t center, cell_t east, 
    cell_t southWest, cell_t south,  cell_t southEast
)
{
    int sum = northWest + north + northEast + west + east + southWest + south + southEast; // get the number of living neighbors
    return (center & ((sum == 2) | (sum == 3))) | (!center & (sum == 3));
}

LifeSimulation::LifeSimulation(int argc, char **argv)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &_worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &_worldSize);

    parseArguments(argc, argv);

    if (_activeProcess) // '-h' or '--help' switch was not specified
    {
        constructMeshTopology();
        if (_activeProcess) // the process is part of the mesh
        {
            readInputFile();
            constructDataTypes();
            constructGridTiles();
        }
    }
}

LifeSimulation::~LifeSimulation()
{
    if (_activeProcess)
    {
        destructGridTiles();
        destructDataTypes();
        destructMeshTopology();
    }
}

void LifeSimulation::parseArguments(int argc, char **argv)
{
    int unused; // to suppress unused return value warning

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
        if (arguments.size() > 1 && (arguments[1] == "-h" || arguments[1] == "--help"))
        {
        help_msg_print:
            if (_worldRank == ROOT)
            {
                cout << "Usage: mpiexec -np <number of processes> ./life <grid file name> <number of iterations> [options]" << endl;
                cout << "Options:" << endl;
                cout << "  -w,   --wraparound                Use wrapped around simulation." << endl;
                cout << "  -nx,  --nodes_x <number>          Number of nodes (processes) in the X (width) direction of the mesh topology." << endl;
                cout << "  -ny,  --nodes_y <number>          Number of nodes (processes) in the Y (height) direction of the mesh topology." << endl;
                cout << "  -p,   --padding <number>          Padding of the global grid in all directions." << endl;
                cout << "  -px,  --padding_x <number>        Padding of the global grid in the X (width) direction." << endl;
                cout << "  -py,  --padding_y <number>        Padding of the global grid in the Y (height) direction." << endl;
                cout << "  -pt,  --padding_top <number>      Padding of the global grid from the top." << endl;
                cout << "  -pb,  --padding_bottom <number>   Padding of the global grid from the bottom." << endl;
                cout << "  -pl,  --padding_left <number>     Padding of the global grid from the left." << endl;
                cout << "  -pr,  --padding_right <number>    Padding of the global grid from the right." << endl;
                cout << "  -ppc, --pixels_per_cell <number>  Number of pixels per cell in the generated images/video." << endl;
                cout << "  -iod, --images_output_directory <directory>" << endl;
                cout << "                                    Directory where generated images will be stored to." << endl;
                cout << "  -v,   --video <file name>         Generate a video of the simulation in mp4 format." << endl;
                cout << "  -fps, --frames_per_second <number>" << endl;
                cout << "                                    Frames per second in the generated video." << endl;
                cout << "  -nfp, --no_formatted_print        Do not print the global grid in a table-like format." << endl;
                cout << "  -ep,  --stderr_print              Print the unformatted global grid to stderr." << endl;
                cout << "  -h,   --help                      Print this help message." << endl;
            }

            _activeProcess = false;
            return;
        }

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
            cerr << "Number of iterations must be a non-negative number." << endl;
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
        else if (*iterator == "-nx" || *iterator == "--nodes_x")
        {
            iterator++;
            if (iterator == arguments.end())
            {
                missingArgumentError(*lastIterator);
            }

            _arguments.nodesWidthCount = parseInt(*iterator, *lastIterator);
        }
        else if (*iterator == "-ny" || *iterator == "--nodes_y")
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
            if (_arguments.paddingTop == 0)
            {
                _arguments.paddingTop = _arguments.padding;
            }
            if (_arguments.paddingBottom == 0)
            {
                _arguments.paddingBottom = _arguments.padding;
            }
            if (_arguments.paddingLeft == 0)
            {
                _arguments.paddingLeft = _arguments.padding;
            }
            if (_arguments.paddingRight == 0)
            {
                _arguments.paddingRight = _arguments.padding;
            }
        }
        else if (*iterator == "-px" || *iterator == "--padding_x")
        {
            iterator++;
            if (iterator == arguments.end())
            {
                missingArgumentError(*lastIterator);
            }

            _arguments.paddingWidth = parseInt(*iterator, *lastIterator);

            // write-through the padding also to the specific paddings if not specified yet
            if (_arguments.paddingLeft == 0)
            {
                _arguments.paddingLeft = _arguments.paddingWidth;
            }
            if (_arguments.paddingRight == 0)
            {
                _arguments.paddingRight = _arguments.paddingWidth;
            }
        }
        else if (*iterator == "-py" || *iterator == "--padding_y")
        {
            iterator++;
            if (iterator == arguments.end())
            {
                missingArgumentError(*lastIterator);
            }

            _arguments.paddingHeight = parseInt(*iterator, *lastIterator);

            // write-through the padding also to the specific paddings if not specified yet
            if (_arguments.paddingTop == 0)
            {
                _arguments.paddingTop = _arguments.paddingHeight;
            }
            if (_arguments.paddingBottom == 0)
            {
                _arguments.paddingBottom = _arguments.paddingHeight;
            }
        }
        else if (*iterator == "-pt" || *iterator == "--padding_top")
        {
            iterator++;
            if (iterator == arguments.end())
            {
                missingArgumentError(*lastIterator);
            }

            _arguments.paddingTop = parseInt(*iterator, *lastIterator);
        }
        else if (*iterator == "-pb" || *iterator == "--padding_bottom")
        {
            iterator++;
            if (iterator == arguments.end())
            {
                missingArgumentError(*lastIterator);
            }

            _arguments.paddingBottom = parseInt(*iterator, *lastIterator);
        }
        else if (*iterator == "-pl" || *iterator == "--padding_left")
        {
            iterator++;
            if (iterator == arguments.end())
            {
                missingArgumentError(*lastIterator);
            }

            _arguments.paddingLeft = parseInt(*iterator, *lastIterator);
        }
        else if (*iterator == "-pr" || *iterator == "--padding_right")
        {
            iterator++;
            if (iterator == arguments.end())
            {
                missingArgumentError(*lastIterator);
            }

            _arguments.paddingRight = parseInt(*iterator, *lastIterator);
        }
        else if (*iterator == "-ppc" || *iterator == "--pixels_per_cell")
        {
            iterator++;
            if (iterator == arguments.end())
            {
                missingArgumentError(*lastIterator);
            }

            _arguments.pixelsPerCell = parseInt(*iterator, *lastIterator);
        }
        else if (*iterator == "-iod" || *iterator == "--images_output_directory")
        {
            iterator++;
            if (iterator == arguments.end())
            {
                missingArgumentError(*lastIterator);
            }

            _arguments.outputImageDirectoryName = *iterator;
            if (_arguments.outputImageDirectoryName.back() != '/')
            {
                _arguments.outputImageDirectoryName += '/';
            }

            if (_worldRank == ROOT)
            {
                unused = system(("mkdir -p " + _arguments.outputImageDirectoryName).c_str());
                (void)unused; // suppress unused warning
            }

            _settings.generateImages = true;
            _settings.generateOnlyVideo = false;
        }
        else if (*iterator == "-v" || *iterator == "--video")
        {
            iterator++;
            if (iterator == arguments.end())
            {
                missingArgumentError(*lastIterator);
            }

            _arguments.videoFileName = *iterator;
            _settings.generateVideo = true;

            if (!_settings.generateImages)
            {
                _arguments.outputImageDirectoryName = "/tmp/life_images/";
                if (_worldRank == ROOT)
                {
                    unused = system(("mkdir -p " + _arguments.outputImageDirectoryName).c_str());
                    unused = system(("rm -f " + _arguments.outputImageDirectoryName + "*.pbm").c_str());
                    (void)unused; // suppress unused warning
                }
                _settings.generateImages = true;
                _settings.generateOnlyVideo = true;
            }
        }
        else if (*iterator == "-fps" || *iterator == "--frames_per_second")
        {
            iterator++;
            if (iterator == arguments.end())
            {
                missingArgumentError(*lastIterator);
            }

            _arguments.fps = parseInt(*iterator, *lastIterator);
        }
        else if (*iterator == "-nfp" || *iterator == "--no_formatted_print")
        {
            _arguments.formattedPrint = false;
        }
        else if (*iterator == "-ep" || *iterator == "--stderr_print")
        {
            _arguments.stderrPrint = true;
        }
        else if (*iterator == "-h" || *iterator == "--help")
        {
            goto help_msg_print;
        }
        else
        {
            if (_worldRank == ROOT)
            {
                #ifndef _TEST_PRINT_
                    cerr << "Warning: Unknown switch '" << *iterator << "'." << endl;
                #endif
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
        else // msb is even
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
        #ifndef _TEST_PRINT_
            cerr << "Warning: Process " << _worldRank << " is not going to be part of the mesh, i.e. will be idle." << endl;
        #endif
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, MPI_UNDEFINED, &_subWorldCommunicator);
        _subWorldCommunicator = MPI_COMM_NULL;
        _meshCommunicator = MPI_COMM_NULL;
        _activeProcess = false;

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
        _subWorldCommunicator = MPI_COMM_NULL;
    }

    if (_meshCommunicator != MPI_COMM_NULL)
    {
        MPI_Comm_free(&_meshCommunicator);
        _meshCommunicator = MPI_COMM_NULL;
    }
}

void LifeSimulation::readInputFile()
{
    if (_meshRank == ROOT) // only the root process reads the input file
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
        _settings.globalWidth = ((_settings.globalNotPaddedWidth + _settings.nodesWidthCount - 1 + _arguments.paddingLeft + _arguments.paddingRight) / 
                                 _settings.nodesWidthCount) * _settings.nodesWidthCount;
        _settings.globalHeight = ((_settings.globalNotPaddedHeight + _settings.nodesHeightCount - 1  + _arguments.paddingTop + _arguments.paddingBottom) / 
                                  _settings.nodesHeightCount) * _settings.nodesHeightCount;
        
        if (_settings.globalWidth != _settings.globalNotPaddedWidth + _arguments.paddingLeft + _arguments.paddingRight) // grid width had to be adjusted
        {
            #ifndef _TEST_PRINT_
                cerr << "Warning: The input file X dimension (width) of " << _settings.globalNotPaddedWidth 
                     << (_arguments.paddingLeft || _arguments.paddingRight ? " with the additional padding " : " ") 
                     << "is not divisible by the X decomposition dimension." << endl;
                cerr << "         The grid X dimension will be extended to " << _settings.globalWidth << "." << endl;
            #endif
        }
        if (_settings.globalHeight != _settings.globalNotPaddedHeight + _arguments.paddingTop + _arguments.paddingBottom) // grid height had to be adjusted
        {
            #ifndef _TEST_PRINT_
                cerr << "Warning: The input file Y dimension (height) of " << _settings.globalNotPaddedHeight 
                     << (_arguments.paddingTop || _arguments.paddingBottom ? " with the additional padding " : " ") 
                     << "is not divisible by the Y decomposition dimension." << endl;
                cerr << "         The grid Y dimension will be extended to " << _settings.globalHeight << "." << endl;
            #endif
        }
        
        // compute the local dimensions of the grid
        _settings.localWidth = _settings.globalWidth / _settings.nodesWidthCount;
        _settings.localHeight = _settings.globalHeight / _settings.nodesHeightCount;
        _settings.localWidthWithHaloZones = _settings.localWidth + 2;
        _settings.localHeightWithHaloZones = _settings.localHeight + 2;
        _settings.localTileSize = _settings.localHeight * _settings.localWidth;
        _settings.localTileSizeWithHaloZones = _settings.localHeightWithHaloZones * _settings.localWidthWithHaloZones;

        // offsets in the global grid of the read data
        int rowLeftPadding = _arguments.paddingLeft;
        int rowRightPadding = _settings.globalWidth - _settings.globalNotPaddedWidth - rowLeftPadding;
        int colTopPadding = _arguments.paddingTop;

        #ifdef _DEBUG_PRINT_
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
        int idx = colTopPadding * _settings.globalWidth;
        do // read rest of the file (1st line is already read)
        {
            idx += rowLeftPadding;
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
            idx += rowRightPadding;
        }
        while (getline(inputFile, row));

        inputFile.close();
    }

    // broadcast the additionally obtained settings to all processes in the mesh
    MPI_Bcast(&_settings, sizeof(Settings), MPI_BYTE, 0, _meshCommunicator);
}

void LifeSimulation::constructDataTypes()
{
    // tile dimensions and position
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

    _tileType = MPI_DATATYPE_NULL;
    _tileResizedType = MPI_DATATYPE_NULL;
    _tileWithHaloZonesType = MPI_DATATYPE_NULL;
    
    _sendHaloZoneTypes[NORTH] = MPI_DATATYPE_NULL;
    _sendHaloZoneTypes[SOUTH] = MPI_DATATYPE_NULL;
    _sendHaloZoneTypes[WEST] = MPI_DATATYPE_NULL;
    _sendHaloZoneTypes[EAST] = MPI_DATATYPE_NULL;

    _recvHaloZoneTypes[NORTH] = MPI_DATATYPE_NULL;
    _recvHaloZoneTypes[SOUTH] = MPI_DATATYPE_NULL;
    _recvHaloZoneTypes[WEST] = MPI_DATATYPE_NULL;
    _recvHaloZoneTypes[EAST] = MPI_DATATYPE_NULL;
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
        _currentTile = nullptr;
    }

    if (_nextTile != nullptr)
    {
        free(_nextTile);
        _nextTile = nullptr;
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
    // initiate corner exchanges, tags are necessary to distinguish between exchanges in wraparound mode
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

    // initiate rest of the halo zones exchange
    MPI_Ineighbor_alltoallw(_nextTile, _neighbourCounts, _neighbourDisplacements, _sendHaloZoneTypes, _nextTile, 
                            _neighbourCounts, _neighbourDisplacements, _recvHaloZoneTypes, _meshCommunicator, &_haloZoneRequest);
}

void LifeSimulation::awaitHaloZonesExchange()
{
    // await corner exchanges
    MPI_Waitall(4, _cornerSendRequests, MPI_STATUSES_IGNORE);
    MPI_Waitall(4, _cornerRecvRequests, MPI_STATUSES_IGNORE);

    // await rest of the halo zones exchange
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
    for (int i = 1; i < _settings.localWidthWithHaloZones - 1; i++)
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

        // accelerate the computation of each row with SIMD instructions
        #pragma omp simd aligned(currentTile, nextTile: 64) simdlen(16)
        for (int j = 2; j < _settings.localWidthWithHaloZones - 2; j++)
        {
            nextTile[centerIdx + j] = updateCell(currentTile[topIdx + j - 1], currentTile[topIdx + j], currentTile[topIdx + j + 1],
                                                 currentTile[centerIdx + j - 1], currentTile[centerIdx + j], currentTile[centerIdx + j + 1],
                                                 currentTile[bottomIdx + j - 1], currentTile[bottomIdx + j], currentTile[bottomIdx + j + 1]);
        }
    }
}

void LifeSimulation::collectLocalTiles()
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
    
        MPI_Barrier(_meshCommunicator); // ensure processes do not print over each other (most of the time at least)
    }
}

void LifeSimulation::unformattedPrintGlobalTile(ostream &stream)
{
    if (_meshRank == ROOT)
    {
        // dump the global tile to stream in ASCII format
        for (int i = 0; i < _settings.globalHeight; i++)
        {
            for (int j = 0; j < _settings.globalWidth; j++)
            {
                stream << static_cast<int>(_globalTile[i * _settings.globalWidth + j]);
            }
            stream << "\n";
        }
    }
}

void LifeSimulation::formattedPrintGlobalTile()
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

    if (_meshRank == ROOT)
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

void LifeSimulation::generateImagePBM()
{
    collectLocalTiles();
    if (_meshRank == ROOT)
    {
        string imageFileName = _arguments.outputImageDirectoryName + to_string(_imageCounter) + ".pbm";
        ofstream outputFile(imageFileName, ios::out);
        if (!outputFile.is_open())
        {
            #ifndef _TEST_PRINT_
                cerr << "Warning: Unable to open image output file '" << imageFileName << "'." << endl;
            #endif
            return;
        }
        _imageCounter++;

        int imageWidth = _settings.globalWidth * _arguments.pixelsPerCell;
        int imageHeight = _settings.globalHeight * _arguments.pixelsPerCell;
        outputFile << "P1" << endl;
        outputFile << imageWidth << " " << imageHeight << endl;

        string row;
        row.resize(imageWidth);
        for (int i = 0; i < _settings.globalHeight; i++)
        {
            for (int j = 0; j < _settings.globalWidth; j++)
            {
                for (int k = 0; k < _arguments.pixelsPerCell; k++)
                {
                    row[j * _arguments.pixelsPerCell + k] = _globalTile[i * _settings.globalWidth + j] ? '1' : '0';
                }
            }

            for (int k = 0; k < _arguments.pixelsPerCell; k++)
            {
                outputFile << row << endl;
            }
        }
    }
}

void LifeSimulation::printFFMPEGCommand()
{
    if (_meshRank == ROOT)
    {
        string outputVideoName = _arguments.outputImageDirectoryName;
        outputVideoName.back() = '.'; // replace '/' with '.'
        outputVideoName += "mp4";
        #ifndef _TEST_PRINT_
            cerr << "To generate a video from the images, run the following command:" << endl;
            cerr << "ffmpeg -framerate " << _arguments.fps << " -i " << _arguments.outputImageDirectoryName << "%d.pbm " << outputVideoName << endl;  
        #endif
    }
}

void LifeSimulation::generateVideoFFMPEG()
{
    if (_meshRank == ROOT)
    {
        string ffmpegCommand = "ffmpeg -y -framerate " + to_string(_arguments.fps) + " -i " + _arguments.outputImageDirectoryName + "%d.pbm " + 
                               _arguments.videoFileName + ">/dev/null 2>&1";
        int result = system(ffmpegCommand.c_str());
        if (result != 0)
        {
            cerr << "Error: Unable to generate video from collected frames to '" << _arguments.videoFileName << "'.\n";
            cerr << "       Check if you have 'ffmpeg' installed (https://ffmpeg.org)." << endl;
        }

        if (!_settings.generateImages)
        {
            string removeTemporaryFiles = "rm -rf " + _arguments.outputImageDirectoryName;
            int unused = system(removeTemporaryFiles.c_str());
            (void)unused; // suppress unused variable warning
        }
    }
}

void LifeSimulation::run()
{
    if (!_activeProcess) // idle processes do not participate in the simulation
    {
        return;
    }

    exchangeInitialData();
    #ifdef _DEBUG_PRINT_
        debugPrintLocalTile();
    #endif

    // simulation loop
    for (int iteration = 0; iteration < _arguments.numberOfIterations; iteration++)
    {   
        computeHaloZones();
        startHaloZonesExchange();
        computeTile();
        awaitHaloZonesExchange();

        if (_settings.generateImages)
        {
            generateImagePBM();
        }

        swap(_currentTile, _nextTile);
    }
    
    collectLocalTiles();

    if (_arguments.stderrPrint)
    {
        unformattedPrintGlobalTile(cerr);
    }

    if (_arguments.formattedPrint)
    {
        formattedPrintGlobalTile();
    }
    else
    {
        unformattedPrintGlobalTile(cout);
    }

    if (_settings.generateVideo)
    {
        generateVideoFFMPEG();
    }

    if (_settings.generateImages && !_settings.generateVideo)
    {
        printFFMPEGCommand();
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    { // RAII scope
        LifeSimulation simulation(argc, argv);
        simulation.run();
    } // ensure all resources are deallocated before MPI_Finalize

    MPI_Finalize();
    return 0;
}
