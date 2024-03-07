/**
 * @file    ParallelHeatSolver.cpp
 *
 * @author  David Mihola <xmihol00@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *          This file contains implementation of parallel heat equation solver
 *          using MPI/OpenMP hybrid approach.
 *
 * @date    2024-02-23
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <ios>
#include <string_view>
#include <iostream>

#include "ParallelHeatSolver.hpp"

using namespace std;

ParallelHeatSolver::ParallelHeatSolver(const SimulationProperties &simulationProps,
                                       const MaterialProperties &materialProps)
    : HeatSolverBase(simulationProps, materialProps)
{
    MPI_Comm_size(MPI_COMM_WORLD, &_worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &_worldRank);

    mSimulationProps.getDecompGrid(_decomposition.nx, _decomposition.ny);

    _edgeSizes.global = mMaterialProps.getEdgeSize();
    _edgeSizes.localWidth = _edgeSizes.global / _decomposition.nx;
    _edgeSizes.localHeight = _edgeSizes.global / _decomposition.ny;

    _offsets.northSouthHalo = 2 * _edgeSizes.localWidth;
    _offsets.westEastHalo = 2 * _edgeSizes.localHeight;

    _simulationHyperParams.airFlowRate = mSimulationProps.getAirflowRate();
    _simulationHyperParams.coolerTemp = mMaterialProps.getCoolerTemperature();

    /**********************************************************************************************************************/
    /*                                  Call init* and alloc* methods in correct order                                    */
    /**********************************************************************************************************************/

    initGridTopology();
    initDataDistribution();
    allocLocalTiles();

    if (!mSimulationProps.getOutputFileName().empty())
    {
        /**********************************************************************************************************************/
        /*                               Open output file if output file name was specified.                                  */
        /*  If mSimulationProps.useParallelIO() flag is set to true, open output file for parallel access, otherwise open it  */
        /*                         only on MASTER rank using sequetial IO. Use openOutputFile* methods.                       */
        /**********************************************************************************************************************/
    }
}

ParallelHeatSolver::~ParallelHeatSolver()
{
    /**********************************************************************************************************************/
    /*                                  Call deinit* and dealloc* methods in correct order                                */
    /*                                             (should be in reverse order)                                           */
    /**********************************************************************************************************************/

    deinitGridTopology();
}

std::string_view ParallelHeatSolver::getCodeType() const
{
    return codeType;
}

void ParallelHeatSolver::initGridTopology()
{
    /**********************************************************************************************************************/
    /*                          Initialize 2D grid topology using non-periodic MPI Cartesian topology.                    */
    /*                       Also create a communicator for middle column average temperature computation.                */
    /**********************************************************************************************************************/

    MPI_Cart_create(MPI_COMM_WORLD, 2, array<int, 2>{_decomposition.ny, _decomposition.nx}.data(), array<int, 2>{false, false}.data(), 0, &_topologyComm);
    MPI_Comm_rank(_topologyComm, &_topologyRank);

    int middleColumn = _decomposition.nx >> 1;
    if (_worldRank % _decomposition.nx == middleColumn) // middle column 
    {
        MPI_Comm_split(MPI_COMM_WORLD, 0, _worldRank / _decomposition.nx, &_middleColComm);
    }
    else // other columns
    {
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, MPI_UNDEFINED, &_middleColComm);
        _middleColComm = MPI_COMM_NULL;
    }
}

void ParallelHeatSolver::deinitGridTopology()
{
    /**********************************************************************************************************************/
    /*      Deinitialize 2D grid topology and the middle column average temperature computation communicator              */
    /**********************************************************************************************************************/

    MPI_Comm_free(&_topologyComm);
    if (_middleColComm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&_middleColComm);
    }
}

void ParallelHeatSolver::initDataDistribution()
{
    /**********************************************************************************************************************/
    /*                 Initialize variables and MPI datatypes for data distribution (float and int).                      */
    /**********************************************************************************************************************/

    if (_decomposition.ny > 1)
    {
        if (_worldRank % _decomposition.nx == 0) // first columns
        {
            MPI_Comm_split(MPI_COMM_WORLD, 0, _worldRank / _decomposition.nx, &_initialScatterColComm);
        }
        else // other columns
        {
            MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, MPI_UNDEFINED, &_initialScatterColComm);
            _initialScatterColComm = MPI_COMM_NULL;
        }
    }
    else
    {
        _initialScatterColComm = MPI_COMM_NULL;
    }

    MPI_Comm_split(MPI_COMM_WORLD, _worldRank / _decomposition.nx, _worldRank % _decomposition.nx, &_initialScatterRowComm);
}

void ParallelHeatSolver::deinitDataDistribution()
{
    /**********************************************************************************************************************/
    /*                       Deinitialize variables and MPI datatypes for data distribution.                              */
    /**********************************************************************************************************************/
    if (_decomposition.ny > 1 && _initialScatterColComm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&_initialScatterColComm);
    }
    MPI_Comm_free(&_initialScatterRowComm);
}

void ParallelHeatSolver::allocLocalTiles()
{
    /**********************************************************************************************************************/
    /*            Allocate local tiles for domain map (1x), domain parameters (1x) and domain temperature (2x).           */
    /*                                               Use AlignedAllocator.                                                */
    /**********************************************************************************************************************/

    _tempTiles[0].resize(_edgeSizes.localWidth * _edgeSizes.localHeight);
    _tempTiles[1].resize(_edgeSizes.localWidth * _edgeSizes.localHeight);
    _domainParamsTile.resize(_edgeSizes.localWidth * _edgeSizes.localHeight);
    _domainMapTile.resize(_edgeSizes.localWidth * _edgeSizes.localHeight);

    _tempHaloZones[0].resize(_edgeSizes.localWidth * 4 + _edgeSizes.localHeight * 4);
    _tempHaloZones[1].resize(_edgeSizes.localWidth * 4 + _edgeSizes.localHeight * 4);
    _domainMapHaloZoneTmp.resize(_edgeSizes.localWidth * 4 + _edgeSizes.localHeight * 4);
    _domainMapHaloZone.resize(_edgeSizes.localWidth * 4 + _edgeSizes.localHeight * 4);
    _domainParamsHaloZoneTmp.resize(_edgeSizes.localWidth * 4 + _edgeSizes.localHeight * 4);
    _domainParamsHaloZone.resize(_edgeSizes.localWidth * 4 + _edgeSizes.localHeight * 4);

    if (_initialScatterColComm != MPI_COMM_NULL)
    {
        _initialScatterTemp.resize(_edgeSizes.global * _edgeSizes.localHeight);
        _initialScatterDomainParams.resize(_edgeSizes.global * _edgeSizes.localHeight);
        _initialScatterDomainMap.resize(_edgeSizes.global * _edgeSizes.localHeight);
    }
}

void ParallelHeatSolver::deallocLocalTiles()
{
    /**********************************************************************************************************************/
    /*                                   Deallocate local tiles (may be empty).                                           */
    /**********************************************************************************************************************/
}

void ParallelHeatSolver::initHaloExchange()
{
    /**********************************************************************************************************************/
    /*                            Initialize variables and MPI datatypes for halo exchange.                               */
    /*                    If mSimulationProps.isRunParallelRMA() flag is set to true, create RMA windows.                 */
    /**********************************************************************************************************************/
}

void ParallelHeatSolver::deinitHaloExchange()
{
    /**********************************************************************************************************************/
    /*                            Deinitialize variables and MPI datatypes for halo exchange.                             */
    /**********************************************************************************************************************/
}

template <typename T>
void ParallelHeatSolver::scatterTiles(const T *globalData, T *localData)
{
    static_assert(std::is_same_v<T, int> || std::is_same_v<T, float>, "Unsupported scatter datatype!");

    /**********************************************************************************************************************/
    /*                      Implement master's global tile scatter to each rank's local tile.                             */
    /*     The template T parameter is restricted to int or float type. You can choose the correct MPI datatype like:     */
    /*                                                                                                                    */
    /*  const MPI_Datatype globalTileType = std::is_same_v<T, int> ? globalFloatTileType : globalIntTileType;             */
    /*  const MPI_Datatype localTileType  = std::is_same_v<T, int> ? localIntTileType    : localfloatTileType;            */
    /**********************************************************************************************************************/
}

template <typename T>
void ParallelHeatSolver::gatherTiles(const T *localData, T *globalData)
{
    static_assert(std::is_same_v<T, int> || std::is_same_v<T, float>, "Unsupported gather datatype!");

    /**********************************************************************************************************************/
    /*                      Implement each rank's local tile gather to master's rank global tile.                         */
    /*     The template T parameter is restricted to int or float type. You can choose the correct MPI datatype like:     */
    /*                                                                                                                    */
    /*  const MPI_Datatype localTileType  = std::is_same_v<T, int> ? localIntTileType    : localfloatTileType;            */
    /*  const MPI_Datatype globalTileType = std::is_same_v<T, int> ? globalFloatTileType : globalIntTileType;             */
    /**********************************************************************************************************************/
}

void ParallelHeatSolver::computeHaloZones(bool current, bool next)
{
    /**********************************************************************************************************************/
    /*  Compute new temperatures in halo zones, so that copy operations can be overlapped with inner region computation.  */
    /*                        Use updateTile method to compute new temperatures in halo zones.                            */
    /*                             TAKE CARE NOT TO COMPUTE THE SAME AREAS TWICE                                          */
    /**********************************************************************************************************************/

    // unpack data into separate pointers for easier access
    // temperature halo zones
    float *tempNorthUpperCurrentHaloZone = _tempHaloZones[current].data();
    float *tempNorthLowerCurrentHaloZone = _tempHaloZones[current].data() + _edgeSizes.localWidth;
    float *tempSouthUpperCurrentHaloZone = _tempHaloZones[current].data() + _offsets.northSouthHalo;
    float *tempSouthLowerCurrentHaloZone = _tempHaloZones[current].data() + _offsets.northSouthHalo + _edgeSizes.localWidth;

    pair<float, float> *tempWestCurrentHaloZones = static_cast<pair<float, float> *>(_tempHaloZones[current].data() + 2 * _offsets.northSouthHalo);
    pair<float, float> *tempEastCurrentHaloZones = static_cast<pair<float, float> *>(_tempHaloZones[current].data() + 2 * _offsets.northSouthHalo + _offsets.westEastHalo);

    float *tempNorthUpperNextHaloZone = _tempHaloZones[next].data();
    float *tempNorthLowerNextHaloZone = _tempHaloZones[next].data() + _edgeSizes.localWidth;
    float *tempSouthUpperNextHaloZone = _tempHaloZones[next].data() + _offsets.northSouthHalo;
    float *tempSouthLowerNextHaloZone = _tempHaloZones[next].data() + _offsets.northSouthHalo + _edgeSizes.localWidth;

    pair<float, float> *tempWestNextHaloZones = static_cast<pair<float, float> *>(_tempHaloZones[next].data() + 2 * _offsets.northSouthHalo);
    pair<float, float> *tempEastNextHaloZones = static_cast<pair<float, float> *>(_tempHaloZones[next].data() + 2 * _offsets.northSouthHalo + _offsets.westEastHalo);

    // temperature tiles
    float *tempTileCurrent = _tempTiles[current].data();
    float *tempTileSouthUpper = _tempTiles[current].data() + _edgeSizes.localWidth;
    float *tempTileSouthLower = _tempTiles[current].data() + _offsets.northSouthHalo;
    float *tempTileNorthUpper = _tempTiles[current].data() + _edgeSizes.localHeight * _edgeSizes.localWidth - _offsets.northSouthHalo;
    float *tempTileNorthLower = _tempTiles[current].data() + _edgeSizes.localHeight * _edgeSizes.localWidth - _edgeSizes.localWidth;

    // domain parameters halo zones
    float *domainParamsNorthUpperHaloZone = _domainParamsHaloZone.data();
    float *domainParamsNorthLowerHaloZone = _domainParamsHaloZone.data() + _edgeSizes.localWidth;
    float *domainParamsSouthUpperHaloZone = _domainParamsHaloZone.data() + _offsets.northSouthHalo;
    float *domainParamsSouthLowerHaloZone = _domainParamsHaloZone.data() + _offsets.northSouthHalo + _edgeSizes.localWidth;

    pair<float, float> *domainParamsWestHaloZones = static_cast<pair<float, float> *>(_domainParamsHaloZone.data() + 2 * _offsets.northSouthHalo);
    pair<float, float> *domainParamsEastHaloZones = static_cast<pair<float, float> *>(_domainParamsHaloZone.data() + 2 * _offsets.northSouthHalo + _offsets.westEastHalo);

    // domain parameters tiles
    float *domainParamsTile = _domainParamsTile.data();
    float *domainParamsTileSouthUpper = _domainParamsTile.data() + _edgeSizes.localWidth;
    float *domainParamsTileSouthLower = _domainParamsTile.data() + _offsets.northSouthHalo;
    float *domainParamsTileNorthUpper = _domainParamsTile.data() + _edgeSizes.localHeight * _edgeSizes.localWidth - _offsets.northSouthHalo;
    float *domainParamsTileNorthLower = _domainParamsTile.data() + _edgeSizes.localHeight * _edgeSizes.localWidth - _edgeSizes.localWidth;

    // domain map
    int *domainMapTile = _domainMapTile.data();

    // left upper corner
    if (!isTopRow() && !isLeftColumn())
    {
        _tempTiles[next][0] = northUpperNextHaloZone[0] = computePoint(
            tempNorthUpperCurrentHaloZone[0], tempNorthLowerCurrentHaloZone[0], tempTileSouthLower[0], tempTileSouthUpper[0],
            tempWestCurrentHaloZones[0].first, tempWestCurrentHaloZones[0].second, tempTileCurrent[1], tempTileCurrent[2], tempTileCurrent[0], 
            domainParamsNorthUpperHaloZone[0], domainParamsNorthLowerHaloZone[0], domainParamsTileSouthLower[0], domainParamsTileSouthUpper[0],
            domainParamsWestHaloZones[0].first, domainParamsWestHaloZones[0].second, domainParamsTile[1], domainParamsTile[2], domainParamsTile[0],
            domainMapTile[0] 
        );
                                                                       
}

void ParallelHeatSolver::startHaloExchangeP2P(float *localData, std::array<MPI_Request, 8> &requests)
{
    /**********************************************************************************************************************/
    /*                       Start the non-blocking halo zones exchange using P2P communication.                          */
    /*                         Use the requests array to return the requests from the function.                           */
    /*                            Don't forget to set the empty requests to MPI_REQUEST_NULL.                             */
    /**********************************************************************************************************************/
}

void ParallelHeatSolver::startHaloExchangeRMA(float *localData, MPI_Win window)
{
    /**********************************************************************************************************************/
    /*                       Start the non-blocking halo zones exchange using RMA communication.                          */
    /*                   Do not forget that you put/get the values to/from the target's opposite side                     */
    /**********************************************************************************************************************/
}

void ParallelHeatSolver::awaitHaloExchangeP2P(std::array<MPI_Request, 8> &requests)
{
    /**********************************************************************************************************************/
    /*                       Wait for all halo zone exchanges to finalize using P2P communication.                        */
    /**********************************************************************************************************************/
}

void ParallelHeatSolver::awaitHaloExchangeRMA(MPI_Win window)
{
    /**********************************************************************************************************************/
    /*                       Wait for all halo zone exchanges to finalize using RMA communication.                        */
    /**********************************************************************************************************************/
}

void ParallelHeatSolver::scatterInitialData()
{
    const float *scatteredTempRow = mMaterialProps.getInitialTemperature().data();
    const float *scatteredDomainParamsRow = mMaterialProps.getDomainParameters().data();
    const int *scatteredDomainMapRow = mMaterialProps.getDomainMap().data();

    // scatter the initial data across the first column nodes
    if (_initialScatterColComm != MPI_COMM_NULL)
    {
        MPI_Scatter(mMaterialProps.getInitialTemperature().data(), _edgeSizes.global * _edgeSizes.localHeight, MPI_FLOAT,
                    _initialScatterTemp.data(), _edgeSizes.global * _edgeSizes.localHeight, MPI_FLOAT, 0, _initialScatterColComm);
        scatteredTempRow = _initialScatterTemp.data();

        MPI_Scatter(mMaterialProps.getDomainParameters().data(), _edgeSizes.global * _edgeSizes.localHeight, MPI_FLOAT,
                    _initialScatterDomainParams.data(), _edgeSizes.global * _edgeSizes.localHeight, MPI_FLOAT, 0, _initialScatterColComm);
        scatteredDomainParamsRow = _initialScatterDomainParams.data();
        
        MPI_Scatter(mMaterialProps.getDomainMap().data(), _edgeSizes.global * _edgeSizes.localHeight, MPI_INT,
                    _initialScatterDomainMap.data(), _edgeSizes.global * _edgeSizes.localHeight, MPI_INT, 0, _initialScatterColComm);
        scatteredDomainMapRow = _initialScatterDomainMap.data();
    }

    // scatter the initial data across each row of nodes from its first column node
    for (size_t i = 0, j = 0; i < _edgeSizes.localHeight * _edgeSizes.global; i += _edgeSizes.global, j += _edgeSizes.localWidth)
    {
        MPI_Scatter(scatteredTempRow + i, _edgeSizes.localWidth, MPI_FLOAT,
                    _tempTiles[0].data() + j, _edgeSizes.localWidth, MPI_FLOAT, 0, _initialScatterRowComm);
        MPI_Scatter(scatteredDomainParamsRow + i, _edgeSizes.localWidth, MPI_FLOAT,
                    _domainParamsTile.data() + j, _edgeSizes.localWidth, MPI_FLOAT, 0, _initialScatterRowComm);
        MPI_Scatter(scatteredDomainMapRow + i, _edgeSizes.localWidth, MPI_INT,
                    _domainMapTile.data() + j, _edgeSizes.localWidth, MPI_INT, 0, _initialScatterRowComm);
    }
}

void ParallelHeatSolver::prepareInitialHaloZones()
{
    // copy to halo zones North and South
    // use the odd halo zone, so that after the initial exchange, the even halo zone can be used for the first iteration
    copy(_tempTiles[0].begin(), _tempTiles[0].begin() + _offsets.northSouthHalo, _tempHaloZones[1].begin()); // North
    copy(_tempTiles[0].end() - _offsets.northSouthHalo, _tempTiles[0].end(), _tempHaloZones[1].begin() + _offsets.northSouthHalo); // South

    copy(_domainMapHaloZoneTmp.begin(), _domainMapHaloZoneTmp.begin() + _offsets.northSouthHalo, _domainMapTile.begin()); // North
    copy(_domainMapHaloZoneTmp.end() - _offsets.northSouthHalo, _domainMapHaloZoneTmp.end(), _domainMapTile.begin() + _offsets.northSouthHalo); // South

    copy(_domainParamsHaloZoneTmp.begin(), _domainParamsHaloZoneTmp.begin() + _offsets.northSouthHalo, _domainParamsTile.begin()); // North
    copy(_domainParamsHaloZoneTmp.end() - _offsets.northSouthHalo, _domainParamsHaloZoneTmp.end(), _domainParamsTile.begin() + _offsets.northSouthHalo); // South

    // copy to halo zones West and East
    for (size_t i = 0; i < _edgeSizes.localHeight; i++)
    {
        // use the odd halo zone, so that after the initial exchange, the even halo zone can be used for the first iteration
        _tempHaloZones[1][2 * _offsets.northSouthHalo + 2 * i] = _tempTiles[0][i * _edgeSizes.localWidth]; // West
        _tempHaloZones[1][2 * _offsets.northSouthHalo + 2 * i + 1] = _tempTiles[0][i * _edgeSizes.localWidth + 1]; // West
        _tempHaloZones[1][2 * _offsets.northSouthHalo + _offsets.westEastHalo + 2 * i] = _tempTiles[0][i * _edgeSizes.localWidth + _edgeSizes.localWidth - 2]; // East
        _tempHaloZones[1][2 * _offsets.northSouthHalo + _offsets.westEastHalo + 2 * i + 1] = _tempTiles[0][i * _edgeSizes.localWidth + _edgeSizes.localWidth - 1]; // East

        _domainMapHaloZoneTmp[2 * _offsets.northSouthHalo + 2 * i] = _domainMapTile[i * _edgeSizes.localWidth]; // West
        _domainMapHaloZoneTmp[2 * _offsets.northSouthHalo + 2 * i + 1] = _domainMapTile[i * _edgeSizes.localWidth + 1]; // West
        _domainMapHaloZoneTmp[2 * _offsets.northSouthHalo + _offsets.westEastHalo + 2 * i] = _domainMapTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 2]; // East
        _domainMapHaloZoneTmp[2 * _offsets.northSouthHalo + _offsets.westEastHalo + 2 * i + 1] = _domainMapTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 1];  // East

        _domainParamsHaloZoneTmp[2 * _offsets.northSouthHalo + 2 * i] = _domainParamsTile[i * _edgeSizes.localWidth]; // West
        _domainParamsHaloZoneTmp[2 * _offsets.northSouthHalo + 2 * i + 1] = _domainParamsTile[i * _edgeSizes.localWidth + 1]; // West
        _domainParamsHaloZoneTmp[2 * _offsets.northSouthHalo + _offsets.westEastHalo + 2 * i] = _domainParamsTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 2]; // East
        _domainParamsHaloZoneTmp[2 * _offsets.northSouthHalo + _offsets.westEastHalo + 2 * i + 1] = _domainParamsTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 1]; // East
    }

    _sendCounts[0] = _sendCounts[1] = _offsets.northSouthHalo;
    _sendCounts[2] = _sendCounts[3] = _offsets.westEastHalo;
    _displacements[0] = 0;
    _displacements[1] = _offsets.northSouthHalo;
    _displacements[2] = 2 * _offsets.northSouthHalo;
    _displacements[3] = 2 * _offsets.northSouthHalo + _offsets.westEastHalo;
}

void ParallelHeatSolver::exchangeInitialHaloZones()
{
    // scatter/gather the halo zones across neighbors 
    MPI_Neighbor_alltoallv(_tempHaloZones[1].data(), _sendCounts, _displacements, MPI_FLOAT,
                           _tempHaloZones[0].data(), _sendCounts, _displacements, MPI_FLOAT, _topologyComm);
    MPI_Neighbor_alltoallv(_domainMapHaloZoneTmp.data(), _sendCounts, _displacements, MPI_INT,
                           _domainMapHaloZone.data(), _sendCounts, _displacements, MPI_INT, _topologyComm);
    MPI_Neighbor_alltoallv(_domainParamsHaloZoneTmp.data(), _sendCounts, _displacements, MPI_FLOAT,
                           _domainParamsHaloZone.data(), _sendCounts, _displacements, MPI_FLOAT, _topologyComm);
}

void ParallelHeatSolver::run(std::vector<float, AlignedAllocator<float>> &outResult)
{
    std::array<MPI_Request, 8> requestsP2P{};

    #if PRINT_DEBUG
        const float *scatteredTempRow = mMaterialProps.getInitialTemperature().data();
        if (_worldRank == 0)
        {
            for (int i = 0; i < _edgeSizes.global; i++)
            {
                for (int j = 0; j < _edgeSizes.global; j++)
                {
                    ((float *)((size_t)(&scatteredTempRow[i * _edgeSizes.global + j])))[0] = (j*4) / _edgeSizes.global + (i / 2 + 5) * 20;
                    cerr << scatteredTempRow[i * _edgeSizes.global + j] << " ";
                }
                cerr << endl;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    #endif

    scatterInitialData();
    prepareInitialHaloZones();
    exchangeInitialHaloZones();   
    
    #if PRINT_DEBUG
        for (int i = 0; i < _worldSize; i++)
        {
            printHalo(i, 0);
        }
    #endif

    // copy initial temperature to the second buffer
    copy(_tempTiles[0].begin(), _tempTiles[0].end(), _tempTiles[1].begin());

    // deallocate the temporary halo zones
    _domainMapHaloZoneTmp.resize(0);
    _domainParamsHaloZoneTmp.resize(0);

    double startTime = MPI_Wtime();

    // 3. Start main iterative simulation loop.
    for (std::size_t iter = 0; iter < mSimulationProps.getNumIterations(); ++iter)
    {
        const std::size_t oldIdx = iter % 2;       // Index of the buffer with old temperatures
        const std::size_t newIdx = (iter + 1) % 2; // Index of the buffer with new temperatures

        /**********************************************************************************************************************/
        /*                            Compute and exchange halo zones using P2P or RMA.                                       */
        /**********************************************************************************************************************/

        /**********************************************************************************************************************/
        /*                           Compute the rest of the tile. Use updateTile method.                                     */
        /**********************************************************************************************************************/

        /**********************************************************************************************************************/
        /*                            Wait for all halo zone exchanges to finalize.                                           */
        /**********************************************************************************************************************/

        if (shouldStoreData(iter))
        {
            /**********************************************************************************************************************/
            /*                          Store the data into the output file using parallel or sequential IO.                      */
            /**********************************************************************************************************************/
        }

        if (shouldPrintProgress(iter) && shouldComputeMiddleColumnAverageTemperature())
        {
            /**********************************************************************************************************************/
            /*                 Compute and print middle column average temperature and print progress report.                     */
            /**********************************************************************************************************************/
        }
    }

    const std::size_t resIdx = mSimulationProps.getNumIterations() % 2; // Index of the buffer with final temperatures

    double elapsedTime = MPI_Wtime() - startTime;

    /**********************************************************************************************************************/
    /*                                     Gather final domain temperature.                                               */
    /**********************************************************************************************************************/

    /**********************************************************************************************************************/
    /*           Compute (sequentially) and report final middle column temperature average and print final report.        */
    /**********************************************************************************************************************/
}

bool ParallelHeatSolver::shouldComputeMiddleColumnAverageTemperature() const
{
    /**********************************************************************************************************************/
    /*                Return true if rank should compute middle column average temperature.                               */
    /**********************************************************************************************************************/

    return false;
}

float ParallelHeatSolver::computeMiddleColumnAverageTemperatureParallel(const float *localData) const
{
    /**********************************************************************************************************************/
    /*                  Implement parallel middle column average temperature computation.                                 */
    /*                      Use OpenMP directives to accelerate the local computations.                                   */
    /**********************************************************************************************************************/

    return 0.f;
}

float ParallelHeatSolver::computeMiddleColumnAverageTemperatureSequential(const float *globalData) const
{
    /**********************************************************************************************************************/
    /*                  Implement sequential middle column average temperature computation.                               */
    /*                      Use OpenMP directives to accelerate the local computations.                                   */
    /**********************************************************************************************************************/

    return 0.f;
}

void ParallelHeatSolver::openOutputFileSequential()
{
    // Create the output file for sequential access.
    mFileHandle = H5Fcreate(mSimulationProps.getOutputFileName(codeType).c_str(),
                            H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (!mFileHandle.valid())
    {
        throw std::ios::failure("Cannot create output file!");
    }
}

void ParallelHeatSolver::storeDataIntoFileSequential(hid_t fileHandle,
                                                     std::size_t iteration,
                                                     const float *globalData)
{
    storeDataIntoFile(fileHandle, iteration, globalData);
}

void ParallelHeatSolver::openOutputFileParallel()
{
#ifdef H5_HAVE_PARALLEL
    Hdf5PropertyListHandle faplHandle{};

    /**********************************************************************************************************************/
    /*                          Open output HDF5 file for parallel access with alignment.                                 */
    /*      Set up faplHandle to use MPI-IO and alignment. The handle will automatically release the resource.            */
    /**********************************************************************************************************************/

    mFileHandle = H5Fcreate(mSimulationProps.getOutputFileName(codeType).c_str(),
                            H5F_ACC_TRUNC,
                            H5P_DEFAULT,
                            faplHandle);
    if (!mFileHandle.valid())
    {
        throw std::ios::failure("Cannot create output file!");
    }
#else
    throw std::runtime_error("Parallel HDF5 support is not available!");
#endif /* H5_HAVE_PARALLEL */
}

void ParallelHeatSolver::storeDataIntoFileParallel(hid_t fileHandle,
                                                   [[maybe_unused]] std::size_t iteration,
                                                   [[maybe_unused]] const float *localData)
{
    if (fileHandle == H5I_INVALID_HID)
    {
        return;
    }

#ifdef H5_HAVE_PARALLEL
    std::array gridSize{static_cast<hsize_t>(mMaterialProps.getEdgeSize()),
                        static_cast<hsize_t>(mMaterialProps.getEdgeSize())};

    // Create new HDF5 group in the output file
    std::string groupName = "Timestep_" + std::to_string(iteration / mSimulationProps.getWriteIntensity());

    Hdf5GroupHandle groupHandle(H5Gcreate(fileHandle, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    {
        /**********************************************************************************************************************/
        /*                                Compute the tile offsets and sizes.                                                 */
        /*               Note that the X and Y coordinates are swapped (but data not altered).                                */
        /**********************************************************************************************************************/

        // Create new dataspace and dataset using it.
        static constexpr std::string_view dataSetName{"Temperature"};

        Hdf5PropertyListHandle datasetPropListHandle{};

        /**********************************************************************************************************************/
        /*                            Create dataset property list to set up chunking.                                        */
        /*                Set up chunking for collective write operation in datasetPropListHandle variable.                   */
        /**********************************************************************************************************************/

        Hdf5DataspaceHandle dataSpaceHandle(H5Screate_simple(2, gridSize.data(), nullptr));
        Hdf5DatasetHandle dataSetHandle(H5Dcreate(groupHandle, dataSetName.data(),
                                                  H5T_NATIVE_FLOAT, dataSpaceHandle,
                                                  H5P_DEFAULT, datasetPropListHandle,
                                                  H5P_DEFAULT));

        Hdf5DataspaceHandle memSpaceHandle{};

        /**********************************************************************************************************************/
        /*                Create memory dataspace representing tile in the memory (set up memSpaceHandle).                    */
        /**********************************************************************************************************************/

        /**********************************************************************************************************************/
        /*              Select inner part of the tile in memory and matching part of the dataset in the file                  */
        /*                           (given by position of the tile in global domain).                                        */
        /**********************************************************************************************************************/

        Hdf5PropertyListHandle propListHandle{};

        /**********************************************************************************************************************/
        /*              Perform collective write operation, writting tiles from all processes at once.                        */
        /*                                   Set up the propListHandle variable.                                              */
        /**********************************************************************************************************************/

        H5Dwrite(dataSetHandle, H5T_NATIVE_FLOAT, memSpaceHandle, dataSpaceHandle, propListHandle, localData);
    }

    {
        // 3. Store attribute with current iteration number in the group.
        static constexpr std::string_view attributeName{"Time"};
        Hdf5DataspaceHandle dataSpaceHandle(H5Screate(H5S_SCALAR));
        Hdf5AttributeHandle attributeHandle(H5Acreate2(groupHandle, attributeName.data(),
                                                       H5T_IEEE_F64LE, dataSpaceHandle,
                                                       H5P_DEFAULT, H5P_DEFAULT));
        const double snapshotTime = static_cast<double>(iteration);
        H5Awrite(attributeHandle, H5T_IEEE_F64LE, &snapshotTime);
    }
#else
    throw std::runtime_error("Parallel HDF5 support is not available!");
#endif /* H5_HAVE_PARALLEL */
}
