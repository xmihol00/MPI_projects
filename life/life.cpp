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
#include <bit>

using namespace std;

#if 0
    #define INFO_PRINT(rank, message) if (rank == 0) { cerr << "Info: " << message << endl; }
#else
    #define INFO_PRINT(rank, message) 
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

    struct Sizes
    {
        int globalHeight;
        int globalWidth;
        int localHeight;
        int localWidth;
        int localHeightWithHaloZones;
        int localWidthWithHaloZones;
        int nodesEdgeCount;
    } _sizes;

    uint8_t *_globalTile;
    uint8_t *_tiles[2];
};

LifeSimulation::LifeSimulation(int argc, char **argv)
{
    parseArguments(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &_worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &_worldSize);

    initializeGridTopology();
    initializeDataTypes();
    initializeGrid();
    performInitialScatter();
}

LifeSimulation::~LifeSimulation()
{
    
}

LifeSimulation::parseArguments(int argc, char **argv)
{
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

LifeSimulation::initializeGridTopology()
{
    int msbPosition = sizeof(int) * 8 - 1 - countl_zero(_worldSize);
    if (mbs & 1) // msb is odd
    {
        msbPosition--; // ensure we have number of processes that has a integral square root
    }

    if (msbPosition < 2 && _worldRank == 0)
    {
        cerr << "Error: The number of processes must be at least 4." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int color = _worldRank >= (1 << msbPosition); // exclude the processes that will not fit on the grid
    if (color)
    {
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, MPI_UNDEFINED, &_subWorldComm);
    }
    else
    {
        MPI_Comm_split(MPI_COMM_WORLD, color, _worldRank, &_subWorldComm);
    }

    _sizes.nodesEdgeCount = msbPosition >> 1;
    int dimensions = 1 << _sizes.nodesEdgeCount;
    MPI_Cart_create(_subWorldComm, 2, {dimensions, dimensions}, {0, 0}, 0, &_gridComm);
}

LifeSimulation::readInputFile()
{
    if (_worldRank == 0)
    {
        ifstream inputFile(_inputFileName);
        if (!inputFile.is_open())
        {
            cerr << "Error: Unable to open input file '" << _inputFileName << "'." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        string row;
        getline(inputFile, row);
        _sizes.globalWidth = row.length();
        _sizes.globalHeight = _sizes.globalWidth;
        int readLines = 1;
        do
        {

        }
        while (getline(inputFile, row) && readLines++ < _sizes.globalHeight);

        if (readLines < _sizes.globalHeight)
        {
            cerr << "Error: The input file does not contain enough data, number of rows must be equal to number of columns. " << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        inputFile.close();
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    LifeSimulation simulation(argc, argv);
    simulation.run();

    MPI_Finalize();
    return 0;
}
