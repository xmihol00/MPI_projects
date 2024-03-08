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
#include <iomanip>

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
     * @brief Initialize variables and MPI datatypes for data scattering and gathering.
     */
    void initDataDistribution();

    /**
     * @brief Deinitialize variables and MPI datatypes for data scattering and gathering.
     */
    void deinitDataDistribution();

    /**
     * @brief Allocate memory for local tiles.
     */
    void allocLocalTiles();

    /**
     * @brief Deallocate memory for local tiles.
     */
    void deallocLocalTiles();

    /**
     * @brief Initialize variables and MPI datatypes for halo exchange.
     */
    void initHaloExchange();

    /**
     * @brief Deinitialize variables and MPI datatypes for halo exchange.
     */
    void deinitHaloExchange();

    /**
     * @brief Scatter global data to local tiles.
     * @tparam T Type of the data to be scattered. Must be either float or int.
     * @param globalData Global data to be scattered.
     * @param localData  Local data to be filled with scattered values.
     */
    template <typename T>
    void scatterTiles(const T *globalData, T *localData);

    /**
     * @brief Gather local tiles to global data.
     * @tparam T Type of the data to be gathered. Must be either float or int.
     * @param localData  Local data to be gathered.
     * @param globalData Global data to be filled with gathered values.
     */
    template <typename T>
    void gatherTiles(const T *localData, T *globalData);

    /**
     * @brief Compute temperature of the next iteration in the halo zones.
     * @param current index of the current temperature values.
     * @param next    index of the next temperature values.
     */
    void computeHaloZones(bool current, bool next);

    /**
     * @brief Compute temperature of the next iteration in the local tiles.
     * @param current index of the current temperature values.
     * @param next    index of the next temperature values.
     */
    void computeTiles(bool current, bool next);

    /**
     * @brief Start halo exchange using point-to-point communication.
     */
    void startHaloExchangeP2P();

    /**
     * @brief Await halo exchange using point-to-point communication.
     */
    void awaitHaloExchangeP2P();

    /**
     * @brief Start halo exchange using RMA communication.
     * @param localData Local data to be exchanged.
     * @param window    MPI_Win object to be used for RMA communication.
     */
    void startHaloExchangeRMA(float *localData, MPI_Win window);

    /**
     * @brief Await halo exchange using RMA communication.
     * @param window MPI_Win object to be used for RMA communication.
     */
    void awaitHaloExchangeRMA(MPI_Win window);

    /**
     * @brief Computes global average temperature of middle column across
     *        processes in "mGridMiddleColComm" communicator.
     *        NOTE: All ranks in the communicator *HAVE* to call this method.
     * @param localData Data of the local tile.
     * @return Returns average temperature over middle of all tiles in the communicator.
     */
    float computeMiddleColumnAverageTemperatureParallel(const float *localData) const;

    /**
     * @brief Computes global average temperature of middle column of the domain
     *        using values collected to MASTER rank.
     *        NOTE: Only single RANK needs to call this method.
     * @param globalData Simulation state collected to the MASTER rank.
     * @return Returns the average temperature.
     */
    float computeMiddleColumnAverageTemperatureSequential(const float *globalData) const;

    /**
     * @brief Opens output HDF5 file for sequential access by MASTER rank only.
     *        NOTE: Only MASTER (rank = 0) should call this method.
     */
    void openOutputFileSequential();

    /**
     * @brief Stores current state of the simulation into the output file.
     *        NOTE: Only MASTER (rank = 0) should call this method.
     * @param fileHandle HDF5 file handle to be used for the writting operation.
     * @param iteration  Integer denoting current iteration number.
     * @param data       Square 2D array of edgeSize x edgeSize elements containing
     *                   simulation state to be stored in the file.
     */
    void storeDataIntoFileSequential(hid_t fileHandle, std::size_t iteration, const float *globalData);

    /**
     * @brief Opens output HDF5 file for parallel/cooperative access.
     *        NOTE: This method *HAS* to be called from all processes in the communicator.
     */
    void openOutputFileParallel();

    /**
     * @brief Stores current state of the simulation into the output file.
     *        NOTE: All processes which opened the file HAVE to call this method collectively.
     * @param fileHandle HDF5 file handle to be used for the writting operation.
     * @param iteration  Integer denoting current iteration number.
     * @param localData  Local 2D array (tile) of mLocalTileSize[0] x mLocalTileSize[1] elements
     *                   to be stored at tile specific position in the output file.
     *                   This method skips halo zones of the tile and stores only relevant data.
     */
    void storeDataIntoFileParallel(hid_t fileHandle, std::size_t iteration, const float *localData);

    /**
     * @brief Determines if the process should compute average temperature of the middle column.
     * @return Returns true if the process should compute average temperature of the middle column.
     */
    bool shouldComputeMiddleColumnAverageTemperature() const;

    void scatterInitialData();

    void prepareInitialHaloZones();

    void exchangeInitialHaloZones();

    inline constexpr bool isTopRow();

    inline constexpr bool isBottomRow();

    inline constexpr bool isLeftColumn();

    inline constexpr bool isRightColumn();

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
    int _rowRank;

    /// @brief Total number of processes in MPI_COMM_WORLD.
    int _worldSize;

    /// @brief Output file handle (parallel or sequential).
    Hdf5FileHandle mFileHandle{};

    MPI_Comm _topologyComm;
    MPI_Comm _middleColComm;
    MPI_Comm _initialScatterColComm;
    MPI_Comm _initialScatterRowComm;

    MPI_Request _haloExchangeRequest;

    struct Decomposition
    {
        int nx;
        int ny;
    } _decomposition;

    struct Sizes
    {
        size_t global;
        size_t localHeight;
        size_t localWidth;
    } _edgeSizes;

    struct Offsets
    {
        size_t northSouthHalo;
        size_t westEastHalo;
    } _offsets;

    struct SimulationHyperParams
    {
        const float airFlowRate;
        const float coolerTemp;
    } _simulationHyperParams;

    std::vector<float, AlignedAllocator<float>> _tempTiles[2];
    std::vector<float, AlignedAllocator<float>> _domainParamsTile;
    std::vector<int, AlignedAllocator<int>> _domainMapTile;

    std::vector<float, AlignedAllocator<float>> _tempHaloZones[2];
    std::vector<float, AlignedAllocator<float>> _domainParamsHaloZoneTmp;
    std::vector<float, AlignedAllocator<float>> _domainParamsHaloZone;

    std::vector<float, AlignedAllocator<float>> _initialScatterTemp;
    std::vector<float, AlignedAllocator<float>> _initialScatterDomainParams;
    std::vector<int, AlignedAllocator<int>> _initialScatterDomainMap;

    // parameters for all to all gather
    int _transferCounts[4] = {0, };
    int _displacements[4] = {0, };

    #define PRINT_DEBUG 1
    #define MANIPULATE_TEMP 0

    #if PRINT_DEBUG
        void printTile(int rank, int tile)
        {
            if (_worldRank == rank)
            {
                std::cerr << "Rank " << _worldRank << " tile:" << std::endl;
                std::cerr << std::setprecision(7) << std::fixed;
                std::cerr << std::setw(9);
                for (int i = 0; i < _edgeSizes.localHeight; i++)
                {
                    for (int j = 0; j < _edgeSizes.localWidth; j++)
                    {
                        std::cerr << _tempTiles[tile][i * _edgeSizes.localWidth + j] << " ";
                    }
                    std::cerr << std::endl;
                }
                std::cerr << std::endl;
                std::cerr << std::flush;
            }

            std::cerr << std::flush;
            MPI_Barrier(MPI_COMM_WORLD);
        }

        void printDomainMap(int rank)
        {
            if (_worldRank == rank)
            {
                std::cerr << std::setprecision(7) << std::fixed;
                std::cerr << "Rank " << _worldRank << " domain map:" << std::endl;
                for (int i = 0; i < _edgeSizes.localHeight; i++)
                {
                    for (int j = 0; j < _edgeSizes.localWidth; j++)
                    {
                        std::cerr << _domainMapTile[i * _edgeSizes.localWidth + j] << " ";
                    }
                    std::cerr << std::endl;
                }
                std::cerr << std::endl;
                std::cerr << std::flush;
            }

            std::cerr << std::flush;
            MPI_Barrier(MPI_COMM_WORLD);
        }

        void printHalo(int rank, int zone)
        {
            if (_worldRank == rank)
            {
                std::cerr << std::setprecision(7) << std::fixed;
                std::cerr << "Rank " << _worldRank << " halo:" << std::endl;
                for (int i = 0; i < 2; i++)
                {
                    for (int j = 0; j < _edgeSizes.localWidth; j++)
                    {
                        std::cerr << _tempHaloZones[zone][i * _edgeSizes.localWidth + j] << " ";
                    }
                    std::cerr << std::endl;
                }
                for (int i = 0; i < 2 * _edgeSizes.localHeight; i += 2)
                {
                    std::cerr << _tempHaloZones[zone][4 * _edgeSizes.localWidth + i] << " " <<  _tempHaloZones[zone][4 * _edgeSizes.localWidth + i + 1] << "    " <<
                        _tempHaloZones[zone][4 * _edgeSizes.localWidth + 2 * _edgeSizes.localHeight + i] << " " << _tempHaloZones[zone][4 * _edgeSizes.localWidth + 2 * _edgeSizes.localHeight + i + 1] << std::endl; 
                }
                for (int i = 2; i < 4; i++)
                {
                    for (int j = 0; j < _edgeSizes.localWidth; j++)
                    {
                        std::cerr << _tempHaloZones[zone][i * _edgeSizes.localWidth + j] << " ";
                    }
                    std::cerr << std::endl;
                }
                std::cerr << std::endl;
                std::cerr << std::flush;
            }

            std::cerr << std::flush;
            MPI_Barrier(MPI_COMM_WORLD);
        }

        bool _print = false;
    #endif
};

inline constexpr bool ParallelHeatSolver::isTopRow()
{
    return _worldRank < _decomposition.nx;
}

inline constexpr bool ParallelHeatSolver::isBottomRow()
{
    return _worldRank >= _worldSize - _decomposition.nx;
}

inline constexpr bool ParallelHeatSolver::isLeftColumn()
{
    return _rowRank == 0;
}

inline constexpr bool ParallelHeatSolver::isRightColumn()
{
    return _rowRank == _decomposition.nx - 1;
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
    if (_print)
    {
        // cerr all input parameters
        std::cerr << _worldRank << " - tempNorthUpper: " << tempNorthUpper << std::endl;
        std::cerr << _worldRank << " - tempNorthLower: " << tempNorthLower << std::endl;
        std::cerr << _worldRank << " - tempSouthLower: " << tempSouthLower << std::endl;
        std::cerr << _worldRank << " - tempSouthUpper: " << tempSouthUpper << std::endl;
        std::cerr << _worldRank << " - tempWestLeft: " << tempWestLeft << std::endl;
        std::cerr << _worldRank << " - tempWestRight: " << tempWestRight << std::endl;
        std::cerr << _worldRank << " - tempEastRight: " << tempEastRight << std::endl;
        std::cerr << _worldRank << " - tempEastLeft: " << tempEastLeft << std::endl;
        std::cerr << _worldRank << " - tempCenter: " << tempCenter << std::endl;
        std::cerr << _worldRank << " - domainParamNorthUpper: " << domainParamNorthUpper << std::endl;
        std::cerr << _worldRank << " - domainParamNorthLower: " << domainParamNorthLower << std::endl;
        std::cerr << _worldRank << " - domainParamSouthLower: " << domainParamSouthLower << std::endl;
        std::cerr << _worldRank << " - domainParamSouthUpper: " << domainParamSouthUpper << std::endl;
        std::cerr << _worldRank << " - domainParamWestLeft: " << domainParamWestLeft << std::endl;
        std::cerr << _worldRank << " - domainParamWestRight: " << domainParamWestRight << std::endl;
        std::cerr << _worldRank << " - domainParamEastRight: " << domainParamEastRight << std::endl;
        std::cerr << _worldRank << " - domainParamEastLeft: " << domainParamEastLeft << std::endl;
        std::cerr << _worldRank << " - domainParamCenter: " << domainParamCenter << std::endl;
        std::cerr << _worldRank << " - domainMapCenter: " << domainMapCenter << std::endl;
    }

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
