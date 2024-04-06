/**
 * @file    ParallelHeatSolver.hpp
 *
 * @author  David Mihola <xmihol00@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *          This file contains implementation of parallel heat equation solver
 *          using MPI/OpenMP hybrid approach.
 *
 * @date    2024-02-23
 */

#ifndef PARALLEL_HEAT_SOLVER_HPP
#define PARALLEL_HEAT_SOLVER_HPP

#include <array>
#include <cstddef>
#include <string_view>
#include <vector>
#include <chrono>
#include <mpi.h>

#include "AlignedAllocator.hpp"
#include "Hdf5Handle.hpp"
#include "HeatSolverBase.hpp"

#define DATA_TYPE_EXCHANGE (0)  // 1 - use MPI datatypes for halo exchange (UNSAFE), 0 - use raw data for halo exchange
#define RAW_EXCHANGE (!DATA_TYPE_EXCHANGE)

#define MEASURE_HALO_ZONE_COMPUTATION_TIME (0) // 1 - measure halo zone computation time, 0 - do not measure halo zone computation time

/**
 * @brief The ParallelHeatSolver class implements parallel MPI based heat
 *        equation solver in 2D using 2D block grid decomposition.
 */
class ParallelHeatSolver : public HeatSolverBase
{
public:
    /**
     * @brief Constructor - Initializes the solver. This includes:
     *        - Construct 2D grid of tiles.
     *        - Create MPI datatypes used in the simulation.
     *        - Open SEQUENTIAL or PARALLEL HDF5 file.
     *        - Allocate data for local tiles.
     *
     * @param simulationProps Parameters of simulation - passed into base class.
     * @param materialProps   Parameters of material - passed into base class.
     */
    ParallelHeatSolver(const SimulationProperties &simulationProps, const MaterialProperties &materialProps);

    /// @brief Inherit constructors from the base class.
    using HeatSolverBase::HeatSolverBase;

    /**
     * @brief Destructor - Releases all resources allocated by the solver.
     */
    virtual ~ParallelHeatSolver() override;

    /// @brief Inherit assignment operator from the base class.
    using HeatSolverBase::operator=;

    /**
     * @brief Run main simulation loop.
     * @param outResult Output array which is to be filled with computed temperature values.
     *                  The vector is pre-allocated and its size is given by dimensions
     *                  of the input file (edgeSize*edgeSize).
     *                  NOTE: The vector is allocated (and should be used) *ONLY*
     *                        by master process (rank 0 in MPI_COMM_WORLD)!
     */
    virtual void run(std::vector<float, AlignedAllocator<float>> &outResult) override;

protected:
private:
    enum { NORTH = 0, SOUTH, WEST, EAST };
    enum class Corners { LEFT_UPPER, RIGHT_UPPER, LEFT_LOWER, RIGHT_LOWER };
    enum class CornerPoints { LEFT_UPPER, RIGHT_UPPER, LEFT_LOWER, RIGHT_LOWER };

    /**
     * @brief Get type of the code.
     * @return Returns type of the code.
     */
    std::string_view getCodeType() const override;

    /**
     * @brief Initialize the grid topology.
     */
    void initGridTopology();

    /**
     * @brief Deinitialize the grid topology.
     */
    void deinitGridTopology();

    /**
     * @brief Initializes MPI datatypes used in the simulation.
     */
    void initDataTypes();

    /**
     * @brief Deinitializes MPI datatypes used in the simulation.
     */
    void deinitDataTypes();

    /**
     * @brief Allocate memory for local tiles.
     */
    void allocLocalTiles();

    /**
     * @brief Compute temperature of the next iteration in the halo zones.
     * @param current index of the current temperature values.
     * @param next    index of the next temperature values.
     */
    void computeTempHaloZones_Raw(bool current, bool next);

    /**
     * @brief Compute temperature of the next iteration in the halo zones.
     * @param current index of the current temperature values.
     * @param next    index of the next temperature values.
     */
    void computeTempHaloZones_DataType(bool current, bool next);

    /**
     * @brief Compute temperature of the next iteration in the local tiles.
     * @param current index of the current temperature values.
     * @param next    index of the next temperature values.
     */
    void computeTempTile_Raw(bool current, bool next);
    void computeTempTile_DataType(bool current, bool next);

    /**
     * @brief Compute the average temperature of the middle column per process, 
     *        reduce it to the middle column root and print progress report.
     * @param iteration Current iteration number.
     */
    void computeAndPrintMidColAverageParallel_Raw(size_t iteration);
    void computeAndPrintMidColAverageParallel_DataType(size_t iteration);

    /**
     * @brief Compute the average temperature of the middle column in the final 
     *        temperature data and print the final report.
     * @param timeElapsed Time elapsed since the start of the simulation.
     * @param outResult   Output array filled with final temperature data.
     */
    void computeAndPrintMidColAverageSequential(float timeElapsed, const std::vector<float, AlignedAllocator<float>> &outResult);

    /**
     * @brief Start halo exchange using point-to-point communication.
     * @param tile When true, tile 1 is the data source, otherwise tile 0 is the data source.
     */
    void startHaloExchangeP2P_Raw();
    void startHaloExchangeP2P_DataType(bool tile);

    /**
     * @brief Await halo exchange using point-to-point communication.
     */
    void awaitHaloExchangeP2P_Raw();
    void awaitHaloExchangeP2P_DataType(bool unused);

    /**
     * @brief Start halo exchange using RMA communication.
     * @param window_tile When true, tile 1 is the data source/widnow 1 is the destination, otherwise tile 0 is the data source/window 0 is the destination.
     */
    void startHaloExchangeRMA_Raw();
    void startHaloExchangeRMA_DataType(bool window_tile);

    /**
     * @brief Await halo exchange using RMA communication.
     * @param window When true, window 1 is the source, otherwise window 0 is the source.
     */
    void awaitHaloExchangeRMA_Raw();
    void awaitHaloExchangeRMA_DataType(bool window);

    /**
     * @brief Opens output HDF5 file for sequential access by MASTER rank only.
     *        NOTE: Only MASTER (rank = 0) should call this method.
     */
    void openOutputFileSequential();

    /**
     * @brief Stores current state of the simulation into the output file.
     *        NOTE: Only MASTER (rank = 0) should call this method.
     * @param iteration  Integer denoting current iteration number.
     * @param data       Square 2D array of edgeSize x edgeSize elements containing
     *                   simulation state to be stored in the file.
     */
    void storeDataIntoFileSequential(std::size_t iteration, const float *globalData);

    /**
     * @brief Opens output HDF5 file for parallel/cooperative access.
     *        NOTE: This method *HAS* to be called from all processes in the communicator.
     */
    void openOutputFileParallel();

    /**
     * @brief Stores current state of the simulation into the output file.
     *        NOTE: All processes which opened the file HAVE to call this method collectively.
     * @param iteration  Integer denoting current iteration number.
     * @param localData  Local 2D array (tile) of mLocalTileSize[0] x mLocalTileSize[1] elements
     *                   to be stored at tile specific position in the output file.
     *                   This method skips halo zones of the tile and stores only relevant data.
     */
    void storeDataIntoFileParallel(std::size_t iteration, int halloOffset, const float *localData);

    /**
     * @brief Determines if the process should compute average temperature of the middle column.
     * @return Returns true if the process should compute average temperature of the middle column.
     */
    bool shouldComputeMiddleColumnAverageTemperature() const;

    /**
     * @brief Splits and sends the domains (temperature, parameters, map) between processes.
     */
    void scatterInitialData_Raw();
    void scatterInitialData_DataType();

    /**
     * @brief Collects the computed temperature data from all processes.
     * @param final     When true, tile 1 is the data source, otherwise tile 0 is the data source.
     * @param outResult Vector to store the collected data to.
     */
    void gatherComputedTempData_Raw(bool tile, std::vector<float, AlignedAllocator<float>> &outResult);
    void gatherComputedTempData_DataType(bool tile, std::vector<float, AlignedAllocator<float>> &outResult);

    /**
     * @brief Determines if a process is not in the top row of the cartesian topology.
     */
    inline constexpr bool isNotTopRow();

    /**
     * @brief Determines if a process is not in the bottom row of the cartesian topology.
     */
    inline constexpr bool isNotBottomRow();

    /**
     * @brief Determines if a process is not in the left column of the cartesian topology.
     */
    inline constexpr bool isNotLeftColumn();

    /**
     * @brief Determines if a process is not in the right column of the cartesian topology.
     */
    inline constexpr bool isNotRightColumn();

    /**
     * @brief Computes the temperature of a point in the domain.
     */
    inline constexpr float computePoint(
        float tempNorthUpper, float tempNorthLower, float tempSouthLower, float tempSouthUpper, 
        float tempWestLeft, float tempWestRight, float tempEastRight, float tempEastLeft, 
        float tempCenter,
        float domainParamNorthUpper, float domainParamNorthLower, float domainParamSouthLower, float domainParamSouthUpper,
        float domainParamWestLeft, float domainParamWestRight, float domainParamEastRight, float domainParamEastLeft, 
        float domainParamCenter,
        int domainMapCenter
    );

    /**
     * @brief Computes the temperature of a corner point in the domain.
     * @param tile  When true, tile 1 is the data source, otherwise tile 0 is the data source.
     */
    template<Corners corner, CornerPoints cornerPoint>
    inline constexpr float computeCornerPoint(bool tile);

    /// @brief Code type string.
    static constexpr std::string_view codeType{"par"};

    /// @brief Size of the halo zone.
    static constexpr std::size_t haloZoneSize{2};

    // MPI process counts and ranks
    /// @brief Process rank in the global communicator (MPI_COMM_WORLD).
    int _worldRank;
    /// @brief Process rank in the middle column communicator (_midColComm).
    int _midColRank;
    /// @brief Total number of processes in MPI_COMM_WORLD.
    int _worldSize;

    /// @brief Output file handle (parallel or sequential).
    Hdf5FileHandle _fileHandle{};

    // communicators
    /// @brief Communicator used for the cartesian topology.
    MPI_Comm _topologyComm;
    /// @brief Communicator used for the middle column temperature averaging.
    MPI_Comm _midColComm;

    /// @brief Request for the halo exchange P2P communication.
    MPI_Request _haloExchangeRequest;

    /// @brief Windows for RMA communication (referred as window 0 and window 1 above).
    MPI_Win _haloExchangeWindows[2];

    // data types
    /// @brief MPI data type for float tiles without halo zones.
    MPI_Datatype _floatTileWithoutHaloZones;
    /// @brief MPI data type for float tiles without halo zones resized to allow scatter/gather.
    MPI_Datatype _floatTileWithoutHaloZonesResized;
    /// @brief MPI data type for float tiles with halo zones.
    MPI_Datatype _floatTileWithHaloZones;

    /// @brief MPI data type for int tiles without halo zones.
    MPI_Datatype _intTileWithoutHaloZones;
    /// @brief MPI data type for int tiles without halo zones resized to allow scatter/gather.
    MPI_Datatype _intTileWithoutHaloZonesResized;
    /// @brief MPI data type for int tiles with halo zones.
    MPI_Datatype _intTileWithHaloZones;

    /// @brief MPI data types send to NORTH, SOUTH, WEST and EAST during P2P neighbour all to all communication and RMA put.
    MPI_Datatype _floatSendHaloZoneTypes[4];
    /// @brief MPI data types received from NORTH, SOUTH, WEST and EAST during P2P neighbour all to all communication.
    MPI_Datatype _floatRecvHaloZoneTypes[4];
    /// @brief MPI data types received from NORTH, SOUTH, WEST and EAST during RMA put.
    MPI_Datatype _floatInverseRecvHaloZoneTypes[4];

    /// @brief number of nodes in each 2D dimension of the decomposition.
    struct Decomposition
    {
        int nx;
        int ny;
    } _decomposition;

    /// @brief sizes of the domain and tiles.
    struct Sizes
    {
        int globalEdge;
        int localHeight;
        int localHeightWithHalos;
        int localWidth;
        int localWidthWithHalos;
        int globalTile;
        int localTile;
        int localTileWithHalos;
        int northSouthHalo;
        int westEastHalo;
    } _sizes;
    
    /// @brief parameters of the simulation.
    struct SimulationHyperParams
    {
        const float airFlowRate;
        const float coolerTemp;
    } _simulationHyperParams;

    // tiles
    /// @brief temperature tiles memory buffers.
    std::vector<float, AlignedAllocator<float>> _tempTiles[2];
    /// @brief domain parameters tiles memory buffer.
    std::vector<float, AlignedAllocator<float>> _domainParamsTile;
    /// @brief domain map tiles memory buffer.
    std::vector<int, AlignedAllocator<int>> _domainMapTile;

    // halo zones
    /// @brief halo zones memory buffers. Unused with Raw data exchange.
    std::vector<float, AlignedAllocator<float>> _tempHaloZones[2];
    /// @brief domain parameters halo zones memory buffer. Used only for initial data exchange.
    std::vector<float, AlignedAllocator<float>> _domainParamsHaloZoneTmp;
    /// @brief domain parameters halo zones memory buffer. Unused with Raw data exchange.
    std::vector<float, AlignedAllocator<float>> _domainParamsHaloZone;

    // parameters for all to all neighbor communication
    /// @brief transfer counts for Raw data exchange.
    int _transferCounts_Raw[4] = {0, };
    /// @brief displacements for Raw data exchange.
    int _displacements_Raw[4] = {0, };
    /// @brief transfer counts for MPI data type data exchange.
    int _transferCounts_DataType[4] = {1, 1, 1, 1};
    /// @brief displacements for MPI data type data exchange.
    MPI_Aint _displacements_DataType[4] = {0, 0, 0, 0};
    /// @brief ranks of neighbors in the cartesian topology.
    int _neighbors[4] = {0, };

    /// @brief inverse displacements for Raw RMA exchange.
    int _inverseDisplacements_Raw[4] = {0, };

    // scatter/gather parameters
    /// @brief scatter/gather counts for initial data exchange and final collection of results.
    std::vector<int> _scatterGatherCounts;
    /// @brief scatter/gather displacements for initial data exchange and final collection of results.
    std::vector<int> _scatterGatherDisplacements;

    #if MEASURE_HALO_ZONE_COMPUTATION_TIME
        size_t _haloZoneComputationDelay = 0;
    #endif
};

inline constexpr bool ParallelHeatSolver::isNotTopRow()
{
    return _neighbors[NORTH] != MPI_PROC_NULL;
}

inline constexpr bool ParallelHeatSolver::isNotBottomRow()
{
    return _neighbors[SOUTH] != MPI_PROC_NULL;
}

inline constexpr bool ParallelHeatSolver::isNotLeftColumn()
{
    return _neighbors[WEST] != MPI_PROC_NULL;
}

inline constexpr bool ParallelHeatSolver::isNotRightColumn()
{
    return _neighbors[EAST] != MPI_PROC_NULL;
}

#pragma omp declare simd notinbranch
inline constexpr float ParallelHeatSolver::computePoint(
    float tempNorthUpper, float tempNorthLower, float tempSouthLower, float tempSouthUpper, 
    float tempWestLeft, float tempWestRight, float tempEastRight, float tempEastLeft, 
    float tempCenter,
    float domainParamNorthUpper, float domainParamNorthLower, float domainParamSouthLower, float domainParamSouthUpper,
    float domainParamWestLeft, float domainParamWestRight, float domainParamEastRight, float domainParamEastLeft, 
    float domainParamCenter,
    int domainMapCenter
)
{
    const float frac = 1.0f / (
        domainParamNorthUpper + domainParamNorthLower + domainParamSouthLower + domainParamSouthUpper + 
        domainParamWestLeft + domainParamWestRight + domainParamEastRight + domainParamEastLeft + domainParamCenter
    );

    float pointTemp = frac * (
        tempNorthUpper * domainParamNorthUpper + 
        tempNorthLower * domainParamNorthLower + 
        tempSouthLower * domainParamSouthLower + 
        tempSouthUpper * domainParamSouthUpper + 
        tempWestLeft * domainParamWestLeft + 
        tempWestRight * domainParamWestRight + 
        tempEastRight * domainParamEastRight + 
        tempEastLeft * domainParamEastLeft + 
        tempCenter * domainParamCenter
    );

    if (domainMapCenter == 0)
    {
        pointTemp = _simulationHyperParams.airFlowRate * _simulationHyperParams.coolerTemp + (1.0f - _simulationHyperParams.airFlowRate) * pointTemp;
    }

    return pointTemp;
}

template<ParallelHeatSolver::Corners corner, ParallelHeatSolver::CornerPoints cornerPoint>
inline constexpr float ParallelHeatSolver::computeCornerPoint(bool tile)
{
    using namespace std;

    pair<float, float> *tempWestHaloZone = reinterpret_cast<pair<float, float> *>(_tempHaloZones[0].data() + 2 * _sizes.northSouthHalo);
    pair<float, float> *tempEastHaloZone = reinterpret_cast<pair<float, float> *>(_tempHaloZones[0].data() + 2 * _sizes.northSouthHalo + _sizes.westEastHalo);
    
    pair<float, float> *domainParamsWestHaloZone = reinterpret_cast<pair<float, float> *>(_domainParamsHaloZone.data() + 2 * _sizes.northSouthHalo);
    pair<float, float> *domainParamsEastHaloZone = reinterpret_cast<pair<float, float> *>(_domainParamsHaloZone.data() + 2 * _sizes.northSouthHalo + _sizes.westEastHalo);

    float tempNorthUpper, tempNorth, tempSouth, tempSouthLower, tempWestLeft, tempWest, tempEast, tempEastRight, tempCenter;
    float domainParamsNorthUpper, domainParamsNorth, domainParamsSouth, domainParamsSouthLower, domainParamsWestLeft, domainParamsWest, domainParamsEast, domainParamsEastRight, domainParamsCenter;
    int domainMapCenter;

    size_t rowCornerOffset = corner == Corners::RIGHT_UPPER || corner == Corners::RIGHT_LOWER ? _sizes.localWidth - 2 : 0;
    size_t rowOffset = cornerPoint == CornerPoints::LEFT_UPPER ? rowCornerOffset :
                            cornerPoint == CornerPoints::RIGHT_UPPER ? rowCornerOffset + 1 :
                                cornerPoint == CornerPoints::LEFT_LOWER ? rowCornerOffset + _sizes.localWidth :
                                    rowCornerOffset + _sizes.localWidth + 1;
    size_t haloZoneCornerOffset = corner == Corners::RIGHT_UPPER || corner == Corners::LEFT_UPPER ? 0 : _sizes.localHeight - 2;
    size_t haloZoneOffset = cornerPoint == CornerPoints::LEFT_UPPER || cornerPoint == CornerPoints::RIGHT_UPPER ? haloZoneCornerOffset : haloZoneCornerOffset + 1;

    if constexpr (corner == Corners::LEFT_UPPER || corner == Corners::RIGHT_UPPER)
    {          
        if constexpr (cornerPoint == CornerPoints::LEFT_UPPER || cornerPoint == CornerPoints::RIGHT_UPPER)
        {
            tempNorthUpper = _tempHaloZones[0][rowOffset];
            tempNorth = _tempHaloZones[0][_sizes.localWidth + rowOffset];
            tempCenter = _tempTiles[tile][rowOffset];
            tempSouth = _tempTiles[tile][_sizes.localWidth + rowOffset];
            tempSouthLower = _tempTiles[tile][2 * _sizes.localWidth + rowOffset];

            domainParamsNorthUpper = _domainParamsHaloZone[rowOffset];
            domainParamsNorth = _domainParamsHaloZone[_sizes.localWidth + rowOffset];
            domainParamsCenter = _domainParamsTile[rowOffset];
            domainParamsSouth = _domainParamsTile[_sizes.localWidth + rowOffset];
            domainParamsSouthLower = _domainParamsTile[2 * _sizes.localWidth + rowOffset];

            domainMapCenter = _domainMapTile[rowOffset];
        }
        else
        {
            tempNorthUpper = _tempHaloZones[0][rowOffset];
            tempNorth = _tempTiles[tile][rowOffset - _sizes.localWidth];
            tempCenter = _tempTiles[tile][rowOffset];
            tempSouth = _tempTiles[tile][_sizes.localWidth + rowOffset];
            tempSouthLower = _tempTiles[tile][2 * _sizes.localWidth + rowOffset];

            domainParamsNorthUpper = _domainParamsHaloZone[rowOffset];
            domainParamsNorth = _domainParamsTile[rowOffset - _sizes.localWidth];
            domainParamsCenter = _domainParamsTile[rowOffset];
            domainParamsSouth = _domainParamsTile[_sizes.localWidth + rowOffset];
            domainParamsSouthLower = _domainParamsTile[2 * _sizes.localWidth + rowOffset];

            domainMapCenter = _domainMapTile[rowOffset];
        }
    }
    else
    {
        const size_t rowHaloZoneOffset = rowOffset + _sizes.northSouthHalo;
        rowOffset += (_sizes.localHeight - 2) * _sizes.localWidth;

        if constexpr (cornerPoint == CornerPoints::LEFT_UPPER || cornerPoint == CornerPoints::RIGHT_UPPER)
        {
            tempNorth = _tempTiles[tile][rowOffset - 2 * _sizes.localWidth];
            tempNorthUpper = _tempTiles[tile][rowOffset - _sizes.localWidth];
            tempCenter = _tempTiles[tile][rowOffset];
            tempSouth = _tempTiles[tile][rowOffset + _sizes.localWidth];
            tempSouthLower = _tempHaloZones[0][rowHaloZoneOffset];

            domainParamsNorth = _domainParamsTile[rowOffset - 2 * _sizes.localWidth];
            domainParamsNorthUpper = _domainParamsTile[rowOffset - _sizes.localWidth];
            domainParamsCenter = _domainParamsTile[rowOffset];
            domainParamsSouth = _domainParamsTile[rowOffset + _sizes.localWidth];
            domainParamsSouthLower = _domainParamsHaloZone[rowHaloZoneOffset];

            domainMapCenter = _domainMapTile[rowOffset];
        }
        else
        {
            tempNorth = _tempTiles[tile][rowOffset - 2 * _sizes.localWidth];
            tempNorthUpper = _tempTiles[tile][rowOffset - _sizes.localWidth];
            tempCenter = _tempTiles[tile][rowOffset];
            tempSouth = _tempHaloZones[0][rowHaloZoneOffset - _sizes.localWidth];
            tempSouthLower = _tempHaloZones[0][rowHaloZoneOffset];

            domainParamsNorth = _domainParamsTile[rowOffset - 2 * _sizes.localWidth];
            domainParamsNorthUpper = _domainParamsTile[rowOffset - _sizes.localWidth];
            domainParamsCenter = _domainParamsTile[rowOffset];
            domainParamsSouth = _domainParamsHaloZone[rowHaloZoneOffset - _sizes.localWidth];
            domainParamsSouthLower = _domainParamsHaloZone[rowHaloZoneOffset];

            domainMapCenter = _domainMapTile[rowOffset];
        }
    }

    if constexpr (corner == Corners::LEFT_UPPER || corner == Corners::LEFT_LOWER)
    {
        if constexpr (cornerPoint == CornerPoints::LEFT_UPPER || cornerPoint == CornerPoints::LEFT_LOWER)
        {
            tempWestLeft = tempWestHaloZone[haloZoneOffset].first;
            tempWest = tempWestHaloZone[haloZoneOffset].second;
            tempEast = _tempTiles[tile][rowOffset + 1];
            tempEastRight = _tempTiles[tile][rowOffset + 2];

            domainParamsWestLeft = domainParamsWestHaloZone[haloZoneOffset].first;
            domainParamsWest = domainParamsWestHaloZone[haloZoneOffset].second;
            domainParamsEast = _domainParamsTile[rowOffset + 1];
            domainParamsEastRight = _domainParamsTile[rowOffset + 2];
        }
        else
        {
            tempWestLeft = tempWestHaloZone[haloZoneOffset].second;
            tempWest = _tempTiles[tile][rowOffset - 1];
            tempEast = _tempTiles[tile][rowOffset + 1];
            tempEastRight = _tempTiles[tile][rowOffset + 2];

            domainParamsWestLeft = domainParamsWestHaloZone[haloZoneOffset].second;
            domainParamsWest = _domainParamsTile[rowOffset - 1];
            domainParamsEast = _domainParamsTile[rowOffset + 1];
            domainParamsEastRight = _domainParamsTile[rowOffset + 2];
        }
    }
    else
    {
        if constexpr (cornerPoint == CornerPoints::LEFT_UPPER || cornerPoint == CornerPoints::LEFT_LOWER)
        {
            tempWestLeft = _tempTiles[tile][rowOffset - 2];
            tempWest = _tempTiles[tile][rowOffset - 1];
            tempEast = _tempTiles[tile][rowOffset + 1];
            tempEastRight = tempEastHaloZone[haloZoneOffset].first;

            domainParamsWestLeft = _domainParamsTile[rowOffset - 2];
            domainParamsWest = _domainParamsTile[rowOffset - 1];
            domainParamsEast = _domainParamsTile[rowOffset + 1];
            domainParamsEastRight = domainParamsEastHaloZone[haloZoneOffset].first;
        }
        else
        {
            tempWestLeft = _tempTiles[tile][rowOffset - 2];
            tempWest = _tempTiles[tile][rowOffset - 1];
            tempEast = tempEastHaloZone[haloZoneOffset].first;
            tempEastRight = tempEastHaloZone[haloZoneOffset].second;

            domainParamsWestLeft = _domainParamsTile[rowOffset - 2];
            domainParamsWest = _domainParamsTile[rowOffset - 1];
            domainParamsEast = domainParamsEastHaloZone[haloZoneOffset].first;
            domainParamsEastRight = domainParamsEastHaloZone[haloZoneOffset].second;
        }
    }

    return computePoint(
        tempNorthUpper, tempNorth, tempSouth, tempSouthLower,
        tempWestLeft, tempWest, tempEast, tempEastRight,
        tempCenter,
        domainParamsNorthUpper, domainParamsNorth, domainParamsSouth, domainParamsSouthLower,
        domainParamsWestLeft, domainParamsWest, domainParamsEast, domainParamsEastRight,
        domainParamsCenter,
        domainMapCenter);
}

#endif /* PARALLEL_HEAT_SOLVER_HPP */
