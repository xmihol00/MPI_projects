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

#include <mpi.h>

#include "AlignedAllocator.hpp"
#include "Hdf5Handle.hpp"
#include "HeatSolverBase.hpp"

#define DATA_TYPE_EXCHANGE (1)  // 0 - use MPI datatypes for halo exchange (UNSAFE), 1 - use raw data for halo exchange
#define RAW_EXCHANGE (!DATA_TYPE_EXCHANGE)

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
     */
    void startHaloExchangeP2P_Raw();
    void startHaloExchangeP2P_DataType(bool next);

    /**
     * @brief Await halo exchange using point-to-point communication.
     */
    void awaitHaloExchangeP2P_Raw();
    void awaitHaloExchangeP2P_DataType(bool next);

    /**
     * @brief Start halo exchange using RMA communication.
     * @param localData Local data to be exchanged.
     * @param window    MPI_Win object to be used for RMA communication.
     */
    void startHaloExchangeRMA_Raw();
    void startHaloExchangeRMA_DataType(bool next);

    /**
     * @brief Await halo exchange using RMA communication.
     * @param window MPI_Win object to be used for RMA communication.
     */
    void awaitHaloExchangeRMA_Raw();
    void awaitHaloExchangeRMA_DataType(bool next);

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

    void scatterInitialData_Raw();
    void scatterInitialData_DataType();

    void gatherComputedTempData_Raw(bool final, std::vector<float, AlignedAllocator<float>> &outResult);
    void gatherComputedTempData_DataType(bool final, std::vector<float, AlignedAllocator<float>> &outResult);

    void prepareInitialHaloZones();

    void exchangeInitialHaloZones();

    void initDataTypes();

    void deinitDataTypes();

    inline constexpr bool isNotTopRow();

    inline constexpr bool isNotBottomRow();

    inline constexpr bool isNotLeftColumn();

    inline constexpr bool isNotRightColumn();

    inline constexpr float computePoint(
        float tempNorthUpper, float tempNorthLower, float tempSouthLower, float tempSouthUpper, 
        float tempWestLeft, float tempWestRight, float tempEastRight, float tempEastLeft, 
        float tempCenter,
        float domainParamNorthUpper, float domainParamNorthLower, float domainParamSouthLower, float domainParamSouthUpper,
        float domainParamWestLeft, float domainParamWestRight, float domainParamEastRight, float domainParamEastLeft, 
        float domainParamCenter,
        int domainMapCenter
    );

    /// @brief Code type string.
    static constexpr std::string_view codeType{"par"};

    /// @brief Size of the halo zone.
    static constexpr std::size_t haloZoneSize{2};

    /// @brief Process rank in the global communicator (MPI_COMM_WORLD).
    int _worldRank;
    int _midColRank;

    /// @brief Total number of processes in MPI_COMM_WORLD.
    int _worldSize;

    /// @brief Output file handle (parallel or sequential).
    Hdf5FileHandle mFileHandle{};

    enum { NORTH = 0, SOUTH, WEST, EAST };

    MPI_Comm _topologyComm;
    MPI_Comm _midColComm;

    MPI_Request _haloExchangeRequest;

    MPI_Win _haloExchangeWindows[2];

    MPI_Datatype _floatTileWithoutHaloZones;
    MPI_Datatype _floatTileWithoutHaloZonesResized;
    MPI_Datatype _floatTileWithHaloZones;

    MPI_Datatype _intTileWithoutHaloZones;
    MPI_Datatype _intTileWithoutHaloZonesResized;
    MPI_Datatype _intTileWithHaloZones;

    MPI_Datatype _floatSendHaloZoneTypes[4];
    MPI_Datatype _floatRecvHaloZoneTypes[4];
    MPI_Datatype _floatInverseRecvHaloZoneTypes[4];

    struct Decomposition
    {
        int nx;
        int ny;
    } _decomposition;

    struct Sizes
    {
        size_t globalEdge;
        size_t localHeight;
        size_t localHeightWithHalos;
        size_t localWidth;
        size_t localWidthWithHalos;
        size_t globalTile;
        size_t localTile;
        size_t localTileWithHalos;
        size_t northSouthHalo;
        size_t westEastHalo;
    } _sizes;
    
    struct SimulationHyperParams
    {
        const float airFlowRate;
        const float coolerTemp;
    } _simulationHyperParams;

    // memory buffers for local tiles
    std::vector<float, AlignedAllocator<float>> _tempTiles[2];
    std::vector<float, AlignedAllocator<float>> _domainParamsTile;
    std::vector<int, AlignedAllocator<int>> _domainMapTile;

    std::vector<float, AlignedAllocator<float>> _tempHaloZones[2];
    std::vector<float, AlignedAllocator<float>> _domainParamsHaloZoneTmp;
    std::vector<float, AlignedAllocator<float>> _domainParamsHaloZone;

    // parameters for scatter and gather
    int _transferCounts_Raw[4] = {0, };
    int _displacements_Raw[4] = {0, };
    int _inverseDisplacements_Raw[4] = {0, };
    int _neighbors[4] = {0, };
    int _transferCounts_DataType[4] = {1, 1, 1, 1};
    MPI_Aint _displacements_DataType[4] = {0, 0, 0, 0};
    std::vector<int> _scatterGatherCounts;
    std::vector<int> _scatterGatherDisplacements;
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

#endif /* PARALLEL_HEAT_SOLVER_HPP */
