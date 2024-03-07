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
    : HeatSolverBase(simulationProps, materialProps), _simulationHyperParams{.airFlowRate = mSimulationProps.getAirflowRate(), 
                                                                             .coolerTemp = mMaterialProps.getCoolerTemperature()}
{
    MPI_Comm_size(MPI_COMM_WORLD, &_worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &_worldRank);

    mSimulationProps.getDecompGrid(_decomposition.nx, _decomposition.ny);

    _edgeSizes.global = mMaterialProps.getEdgeSize();
    _edgeSizes.localWidth = _edgeSizes.global / _decomposition.nx;
    _edgeSizes.localHeight = _edgeSizes.global / _decomposition.ny;

    _offsets.northSouthHalo = 2 * _edgeSizes.localWidth;
    _offsets.westEastHalo = 2 * _edgeSizes.localHeight;

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
    MPI_Comm_set_name(_topologyComm, "Topology Communicator");

    int middleColumn = _decomposition.nx >> 1;
    if (_worldRank % _decomposition.nx == middleColumn) // middle column 
    {
        MPI_Comm_split(MPI_COMM_WORLD, 0, _worldRank / _decomposition.nx, &_middleColComm);
        MPI_Comm_set_name(_topologyComm, "Middle Column Communicator");
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
            MPI_Comm_set_name(_initialScatterColComm, "Initial Scatter Column Communicator");
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
    MPI_Comm_rank(_initialScatterRowComm, &_rowRank);
    MPI_Comm_set_name(_initialScatterRowComm, "Initial Scatter Row Communicator");
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

    // unpack data into separate pointers for easier access, simulate the complete tile with halo zones
    // temperature
    float *tempTopRow0Current = _tempHaloZones[current].data(); // closest to the top
    float *tempTopRow1Current = _tempHaloZones[current].data() + _edgeSizes.localWidth;
    float *tempTopRow2Current = _tempTiles[current].data();
    float *tempTopRow3Current = _tempTiles[current].data() + _edgeSizes.localWidth;
    float *tempTopRow4Current = _tempTiles[current].data() + _offsets.northSouthHalo;
    float *tempTopRow5Current = _tempTiles[current].data() + _offsets.northSouthHalo + _edgeSizes.localWidth;

    float *tempBotRow5Current = _tempTiles[current].data() + _edgeSizes.localHeight * _edgeSizes.localWidth - 2 * _offsets.northSouthHalo;
    float *tempBotRow4Current = _tempTiles[current].data() + _edgeSizes.localHeight * _edgeSizes.localWidth - _offsets.northSouthHalo - _edgeSizes.localWidth;
    float *tempBotRow3Current = _tempTiles[current].data() + _edgeSizes.localHeight * _edgeSizes.localWidth - _offsets.northSouthHalo;
    float *tempBotRow2Current = _tempTiles[current].data() + _edgeSizes.localHeight * _edgeSizes.localWidth - _edgeSizes.localWidth;
    float *tempBotRow1Current = _tempHaloZones[current].data() + _offsets.northSouthHalo;
    float *tempBotRow0Current = _tempHaloZones[current].data() + _offsets.northSouthHalo + _edgeSizes.localWidth; // closest to the bottom

    pair<float, float> *tempWestHaloZoneCurrent = reinterpret_cast<pair<float, float> *>(_tempHaloZones[current].data() + 2 * _offsets.northSouthHalo);
    pair<float, float> *tempEastHaloZoneCurrent = reinterpret_cast<pair<float, float> *>(_tempHaloZones[current].data() + 2 * _offsets.northSouthHalo + _offsets.westEastHalo);

    float *tempHaloTopRow0Next = _tempHaloZones[next].data();
    float *tempHaloTopRow1Next = _tempHaloZones[next].data() + _edgeSizes.localWidth;
    float *tempTileTopRow0Next = _tempTiles[next].data();
    float *tempTileTopRow1Next = _tempTiles[next].data() + _edgeSizes.localWidth;

    float *tempHaloBotRow1Next = _tempHaloZones[next].data() + _offsets.northSouthHalo;
    float *tempHaloBotRow0Next = _tempHaloZones[next].data() + _offsets.northSouthHalo + _edgeSizes.localWidth;
    float *tempTileBotRow1Next = _tempTiles[next].data() + _edgeSizes.localHeight * _edgeSizes.localWidth - _offsets.northSouthHalo;
    float *tempTileBotRow0Next = _tempTiles[next].data() + _edgeSizes.localHeight * _edgeSizes.localWidth - _edgeSizes.localWidth;

    pair<float, float> *tempWestHaloZoneNext = reinterpret_cast<pair<float, float> *>(_tempHaloZones[next].data() + 2 * _offsets.northSouthHalo);
    pair<float, float> *tempEastHaloZoneNext = reinterpret_cast<pair<float, float> *>(_tempHaloZones[next].data() + 2 * _offsets.northSouthHalo + _offsets.westEastHalo);

    float *tempCurrentTile = _tempTiles[current].data();
    float *tempNextTile = _tempTiles[next].data();

    // domain parameters
    float *domainParamsTopRow0 = _domainParamsHaloZone.data();
    float *domainParamsTopRow1 = _domainParamsHaloZone.data() + _edgeSizes.localWidth;
    float *domainParamsTopRow2 = _domainParamsTile.data();
    float *domainParamsTopRow3 = _domainParamsTile.data() + _edgeSizes.localWidth;
    float *domainParamsTopRow4 = _domainParamsTile.data() + _offsets.northSouthHalo;
    float *domainParamsTopRow5 = _domainParamsTile.data() + _offsets.northSouthHalo + _edgeSizes.localWidth;

    float *domainParamsBotRow5 = _domainParamsTile.data() + _edgeSizes.localHeight * _edgeSizes.localWidth - 2 * _offsets.northSouthHalo;
    float *domainParamsBotRow4 = _domainParamsTile.data() + _edgeSizes.localHeight * _edgeSizes.localWidth - _offsets.northSouthHalo - _edgeSizes.localWidth;
    float *domainParamsBotRow3 = _domainParamsTile.data() + _edgeSizes.localHeight * _edgeSizes.localWidth - _offsets.northSouthHalo;
    float *domainParamsBotRow2 = _domainParamsTile.data() + _edgeSizes.localHeight * _edgeSizes.localWidth - _edgeSizes.localWidth;
    float *domainParamsBotRow1 = _domainParamsHaloZone.data() + _offsets.northSouthHalo;
    float *domainParamsBotRow0 = _domainParamsHaloZone.data() + _offsets.northSouthHalo + _edgeSizes.localWidth;    

    pair<float, float> *domainParamsWestHaloZone = reinterpret_cast<pair<float, float> *>(_domainParamsHaloZone.data() + 2 * _offsets.northSouthHalo);
    pair<float, float> *domainParamsEastHaloZone = reinterpret_cast<pair<float, float> *>(_domainParamsHaloZone.data() + 2 * _offsets.northSouthHalo + _offsets.westEastHalo);

    // domain map
    int *domainMapTile = _domainMapTile.data();
    int *domainMapTopCenter0 = _domainMapTile.data();
    int *domainMapTopCenter1 = _domainMapTile.data() + _edgeSizes.localWidth;

    int *domainMapBotCenter1 = _domainMapTile.data() + _edgeSizes.localHeight * _edgeSizes.localWidth - _offsets.northSouthHalo;
    int *domainMapBotCenter0 = _domainMapTile.data() + _edgeSizes.localHeight * _edgeSizes.localWidth - _edgeSizes.localWidth;

    // left upper corner
    if (!isTopRow() && !isLeftColumn()) // node is not in the top row and not in the left column
    {
        tempHaloTopRow0Next[0] = tempTileTopRow0Next[0] = computePoint(
            tempTopRow0Current[0], tempTopRow1Current[0], tempTopRow3Current[0], tempTopRow4Current[0], 
            tempWestHaloZoneCurrent[0].first, tempWestHaloZoneCurrent[0].second, tempTopRow2Current[1], tempTopRow2Current[2],
            tempTopRow2Current[0], 
            domainParamsTopRow0[0], domainParamsTopRow1[0], domainParamsTopRow3[0], domainParamsTopRow4[0],
            domainParamsWestHaloZone[0].first, domainParamsWestHaloZone[0].second, domainParamsTopRow2[1], domainParamsTopRow2[2],
            domainParamsTopRow2[0],
            domainMapTopCenter0[0]
        );
        tempHaloTopRow0Next[1] = tempTileTopRow0Next[1] = computePoint(
            tempTopRow0Current[1], tempTopRow1Current[1], tempTopRow3Current[1], tempTopRow4Current[1], 
            tempWestHaloZoneCurrent[0].second, tempTopRow2Current[0], tempTopRow2Current[2], tempTopRow2Current[3],
            tempTopRow2Current[1],
            domainParamsTopRow0[1], domainParamsTopRow1[1], domainParamsTopRow3[1], domainParamsTopRow4[1],
            domainParamsWestHaloZone[0].second, domainParamsTopRow2[0], domainParamsTopRow2[2], domainParamsTopRow2[3],
            domainParamsTopRow2[1],
            domainMapTopCenter0[1]
        );

        tempHaloTopRow1Next[0] = tempTileTopRow1Next[0] = computePoint(
            tempTopRow1Current[0], tempTopRow2Current[0], tempTopRow4Current[0], tempTopRow5Current[0], 
            tempWestHaloZoneCurrent[1].first, tempWestHaloZoneCurrent[1].second, tempTopRow3Current[1], tempTopRow3Current[2],
            tempTopRow3Current[0],
            domainParamsTopRow1[0], domainParamsTopRow2[0], domainParamsTopRow4[0], domainParamsTopRow5[0],
            domainParamsWestHaloZone[1].first, domainParamsWestHaloZone[1].second, domainParamsTopRow3[1], domainParamsTopRow3[2],
            domainParamsTopRow3[0],
            domainMapTopCenter1[0]
        );
        tempHaloTopRow1Next[1] = tempTileTopRow1Next[1] = computePoint(
            tempTopRow1Current[1], tempTopRow2Current[1], tempTopRow4Current[1], tempTopRow5Current[1], 
            tempWestHaloZoneCurrent[1].second, tempTopRow3Current[0], tempTopRow3Current[2], tempTopRow3Current[3],
            tempTopRow3Current[1],
            domainParamsTopRow1[1], domainParamsTopRow2[1], domainParamsTopRow4[1], domainParamsTopRow5[1],
            domainParamsWestHaloZone[1].second, domainParamsTopRow3[0], domainParamsTopRow3[2], domainParamsTopRow3[3],
            domainParamsTopRow3[1],
            domainMapTopCenter1[1]
        );
    }

    // top row
    if (!isTopRow()) // node is not in the top row
    {
        for (size_t i = 2; i < _edgeSizes.localWidth - 2; i++)
        {
            tempHaloTopRow0Next[i] = tempTileTopRow0Next[i] = computePoint(
                tempTopRow0Current[i], tempTopRow1Current[i], tempTopRow3Current[i], tempTopRow4Current[i], 
                tempTopRow2Current[i - 2], tempTopRow2Current[i - 1], tempTopRow2Current[i + 1], tempTopRow2Current[i + 2],
                tempTopRow2Current[i],
                domainParamsTopRow0[i], domainParamsTopRow1[i], domainParamsTopRow3[i], domainParamsTopRow4[i],
                domainParamsTopRow2[i - 2], domainParamsTopRow2[i - 1], domainParamsTopRow2[i + 1], domainParamsTopRow2[i + 2],
                domainParamsTopRow2[i],
                domainMapTile[i]
            );
            tempHaloTopRow1Next[i] = tempTileTopRow1Next[i] = computePoint(
                tempTopRow1Current[i], tempTopRow2Current[i], tempTopRow4Current[i], tempTopRow5Current[i], 
                tempTopRow3Current[i - 2], tempTopRow3Current[i - 1], tempTopRow3Current[i + 1], tempTopRow3Current[i + 2],
                tempTopRow3Current[i],
                domainParamsTopRow1[i], domainParamsTopRow2[i], domainParamsTopRow4[i], domainParamsTopRow5[i],
                domainParamsTopRow3[i - 2], domainParamsTopRow3[i - 1], domainParamsTopRow3[i + 1], domainParamsTopRow3[i + 2],
                domainParamsTopRow3[i],
                domainMapTile[i]
            );
        }
    }

    // right upper corner
    if (!isTopRow() && !isRightColumn()) // node is not in the top row and not in the right column
    {
        tempHaloTopRow0Next[_edgeSizes.localWidth - 2] = tempTileTopRow0Next[_edgeSizes.localWidth - 2] = computePoint(
            tempTopRow0Current[_edgeSizes.localWidth - 2], tempTopRow1Current[_edgeSizes.localWidth - 2], tempTopRow3Current[_edgeSizes.localWidth - 2], tempTopRow4Current[_edgeSizes.localWidth - 2], 
            tempTopRow2Current[_edgeSizes.localWidth - 4], tempTopRow2Current[_edgeSizes.localWidth - 3], tempTopRow2Current[_edgeSizes.localWidth - 1], tempEastHaloZoneCurrent[_edgeSizes.localHeight - 1].first,
            tempTopRow2Current[_edgeSizes.localWidth - 2],
            domainParamsTopRow0[_edgeSizes.localWidth - 2], domainParamsTopRow1[_edgeSizes.localWidth - 2], domainParamsTopRow3[_edgeSizes.localWidth - 2], domainParamsTopRow4[_edgeSizes.localWidth - 2],
            domainParamsTopRow2[_edgeSizes.localWidth - 4], domainParamsTopRow2[_edgeSizes.localWidth - 3], domainParamsTopRow2[_edgeSizes.localWidth - 1], domainParamsEastHaloZone[_edgeSizes.localHeight - 1].first,
            domainParamsTopRow2[_edgeSizes.localWidth - 2],
            domainMapTile[_edgeSizes.localWidth - 2]
        );
        tempHaloTopRow0Next[_edgeSizes.localWidth - 1] = tempTileTopRow0Next[_edgeSizes.localWidth - 1] = computePoint(
            tempTopRow0Current[_edgeSizes.localWidth - 1], tempTopRow1Current[_edgeSizes.localWidth - 1], tempTopRow3Current[_edgeSizes.localWidth - 1], tempTopRow4Current[_edgeSizes.localWidth - 1], 
            tempTopRow2Current[_edgeSizes.localWidth - 3], tempTopRow2Current[_edgeSizes.localWidth - 2], tempEastHaloZoneCurrent[_edgeSizes.localHeight - 1].first, tempEastHaloZoneCurrent[_edgeSizes.localHeight - 1].second,
            tempTopRow2Current[_edgeSizes.localWidth - 1],
            domainParamsTopRow0[_edgeSizes.localWidth - 1], domainParamsTopRow1[_edgeSizes.localWidth - 1], domainParamsTopRow3[_edgeSizes.localWidth - 1], domainParamsTopRow4[_edgeSizes.localWidth - 1],
            domainParamsTopRow2[_edgeSizes.localWidth - 3], domainParamsTopRow2[_edgeSizes.localWidth - 2], domainParamsEastHaloZone[_edgeSizes.localHeight - 1].first, domainParamsEastHaloZone[_edgeSizes.localHeight - 1].second,
            domainParamsTopRow2[_edgeSizes.localWidth - 1],
            domainMapTile[_edgeSizes.localWidth - 1]
        );

        tempHaloTopRow1Next[_edgeSizes.localWidth - 2] = tempTileTopRow1Next[_edgeSizes.localWidth - 2] = computePoint(
            tempTopRow1Current[_edgeSizes.localWidth - 2], tempTopRow2Current[_edgeSizes.localWidth - 2], tempTopRow4Current[_edgeSizes.localWidth - 2], tempTopRow5Current[_edgeSizes.localWidth - 2], 
            tempTopRow3Current[_edgeSizes.localWidth - 4], tempTopRow3Current[_edgeSizes.localWidth - 3], tempEastHaloZoneCurrent[_edgeSizes.localHeight - 2].first, tempEastHaloZoneCurrent[_edgeSizes.localHeight - 2].second,
            tempTopRow3Current[_edgeSizes.localWidth - 2],
            domainParamsTopRow1[_edgeSizes.localWidth - 2], domainParamsTopRow2[_edgeSizes.localWidth - 2], domainParamsTopRow4[_edgeSizes.localWidth - 2], domainParamsTopRow5[_edgeSizes.localWidth - 2],
            domainParamsTopRow3[_edgeSizes.localWidth - 4], domainParamsTopRow3[_edgeSizes.localWidth - 3], domainParamsEastHaloZone[_edgeSizes.localHeight - 2].first, domainParamsEastHaloZone[_edgeSizes.localHeight - 2].second,
            domainParamsTopRow3[_edgeSizes.localWidth - 2],
            domainMapTile[_edgeSizes.localWidth - 2]
        );
        tempHaloTopRow1Next[_edgeSizes.localWidth - 1] = tempTileTopRow1Next[_edgeSizes.localWidth - 1] = computePoint(
            tempTopRow1Current[_edgeSizes.localWidth - 1], tempTopRow2Current[_edgeSizes.localWidth - 1], tempTopRow4Current[_edgeSizes.localWidth - 1], tempTopRow5Current[_edgeSizes.localWidth - 1], 
            tempTopRow3Current[_edgeSizes.localWidth - 3], tempEastHaloZoneCurrent[_edgeSizes.localHeight - 2].first, tempEastHaloZoneCurrent[_edgeSizes.localHeight - 2].second, tempTopRow3Current[_edgeSizes.localWidth - 1],
            tempTopRow3Current[_edgeSizes.localWidth - 2],
            domainParamsTopRow1[_edgeSizes.localWidth - 1], domainParamsTopRow2[_edgeSizes.localWidth - 1], domainParamsTopRow4[_edgeSizes.localWidth - 1], domainParamsTopRow5[_edgeSizes.localWidth - 1],
            domainParamsTopRow3[_edgeSizes.localWidth - 3], domainParamsEastHaloZone[_edgeSizes.localHeight - 2].first, domainParamsEastHaloZone[_edgeSizes.localHeight - 2].second, domainParamsTopRow3[_edgeSizes.localWidth - 1],
            domainParamsTopRow3[_edgeSizes.localWidth - 2],
            domainMapTile[_edgeSizes.localWidth - 1]
        );
    }

    // columns
    if (!isLeftColumn()) // node is not in the left column
    {
        for (size_t i = 2; i < _edgeSizes.localHeight - 2; i++)
        {
            tempWestHaloZoneNext[i].first = tempNextTile[i * _edgeSizes.localWidth] = computePoint(
                tempCurrentTile[(i - 2) * _edgeSizes.localWidth], tempCurrentTile[(i - 1) * _edgeSizes.localWidth], tempCurrentTile[(i + 1) * _edgeSizes.localWidth], tempCurrentTile[(i + 2) * _edgeSizes.localWidth], 
                tempWestHaloZoneCurrent[i].first, tempWestHaloZoneCurrent[i].second, tempCurrentTile[i * _edgeSizes.localWidth + 1], tempCurrentTile[i * _edgeSizes.localWidth + 2],
                tempCurrentTile[i * _edgeSizes.localWidth],
                _domainParamsTile[(i - 2) * _edgeSizes.localWidth], _domainParamsTile[(i - 1) * _edgeSizes.localWidth], _domainParamsTile[(i + 1) * _edgeSizes.localWidth], _domainParamsTile[(i + 2) * _edgeSizes.localWidth],
                domainParamsWestHaloZone[i].first, domainParamsWestHaloZone[i].second, _domainParamsTile[i * _edgeSizes.localWidth + 1], _domainParamsTile[i * _edgeSizes.localWidth + 2],
                _domainParamsTile[i * _edgeSizes.localWidth],
                _domainMapTile[i * _edgeSizes.localWidth]
            );
            tempWestHaloZoneNext[i].second = tempNextTile[i * _edgeSizes.localWidth + 1] = computePoint(
                tempCurrentTile[(i - 2) * _edgeSizes.localWidth + 1], tempCurrentTile[(i - 1) * _edgeSizes.localWidth + 1], tempCurrentTile[(i + 1) * _edgeSizes.localWidth + 1], tempCurrentTile[(i + 2) * _edgeSizes.localWidth + 1], 
                tempWestHaloZoneCurrent[i].second, tempCurrentTile[i * _edgeSizes.localWidth], tempCurrentTile[i * _edgeSizes.localWidth + 2], tempCurrentTile[i * _edgeSizes.localWidth + 3],
                tempCurrentTile[i * _edgeSizes.localWidth + 1],
                _domainParamsTile[(i - 2) * _edgeSizes.localWidth + 1], _domainParamsTile[(i - 1) * _edgeSizes.localWidth + 1], _domainParamsTile[(i + 1) * _edgeSizes.localWidth + 1], _domainParamsTile[(i + 2) * _edgeSizes.localWidth + 1],
                domainParamsWestHaloZone[i].second, _domainParamsTile[i * _edgeSizes.localWidth], _domainParamsTile[i * _edgeSizes.localWidth + 2], _domainParamsTile[i * _edgeSizes.localWidth + 3],
                _domainParamsTile[i * _edgeSizes.localWidth + 1],
                _domainMapTile[i * _edgeSizes.localWidth + 1]
            );
        }
    }

    if (!isRightColumn()) // node is not in the right column
    {
        for (size_t i = 2; i < _edgeSizes.localHeight - 2; i++)
        {
            tempEastHaloZoneNext[i].first = tempNextTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 2] = computePoint(
                tempCurrentTile[(i - 2) * _edgeSizes.localWidth + _edgeSizes.localWidth - 2], tempCurrentTile[(i - 1) * _edgeSizes.localWidth + _edgeSizes.localWidth - 2], tempCurrentTile[(i + 1) * _edgeSizes.localWidth + _edgeSizes.localWidth - 2], tempCurrentTile[(i + 2) * _edgeSizes.localWidth + _edgeSizes.localWidth - 2],
                tempCurrentTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 4], tempCurrentTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 3], tempCurrentTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 1], tempEastHaloZoneCurrent[i].first,
                tempCurrentTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 2],
                _domainParamsTile[(i - 2) * _edgeSizes.localWidth + _edgeSizes.localWidth - 2], _domainParamsTile[(i - 1) * _edgeSizes.localWidth + _edgeSizes.localWidth - 2], _domainParamsTile[(i + 1) * _edgeSizes.localWidth + _edgeSizes.localWidth - 2], _domainParamsTile[(i + 2) * _edgeSizes.localWidth + _edgeSizes.localWidth - 2],
                _domainParamsTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 4], _domainParamsTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 3], _domainParamsTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 1], domainParamsEastHaloZone[i].first,
                _domainParamsTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 2],
                _domainMapTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 2]
            );
            tempEastHaloZoneNext[i].second = tempNextTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 1] = computePoint(
                tempCurrentTile[(i - 2) * _edgeSizes.localWidth + _edgeSizes.localWidth - 1], tempCurrentTile[(i - 1) * _edgeSizes.localWidth + _edgeSizes.localWidth - 1], tempCurrentTile[(i + 1) * _edgeSizes.localWidth + _edgeSizes.localWidth - 1], tempCurrentTile[(i + 2) * _edgeSizes.localWidth + _edgeSizes.localWidth - 1],
                tempCurrentTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 3], tempCurrentTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 2], tempEastHaloZoneCurrent[i].first, tempEastHaloZoneCurrent[i].second,
                tempCurrentTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 1],
                _domainParamsTile[(i - 2) * _edgeSizes.localWidth + _edgeSizes.localWidth - 1], _domainParamsTile[(i - 1) * _edgeSizes.localWidth + _edgeSizes.localWidth - 1], _domainParamsTile[(i + 1) * _edgeSizes.localWidth + _edgeSizes.localWidth - 1], _domainParamsTile[(i + 2) * _edgeSizes.localWidth + _edgeSizes.localWidth - 1],
                _domainParamsTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 3], _domainParamsTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 2], domainParamsEastHaloZone[i].first, domainParamsEastHaloZone[i].second,
                _domainParamsTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 1],
                _domainMapTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 1]
            );
        }
    }

    // left lower corner
    if (!isBottomRow() && !isLeftColumn()) // node is not in the bottom row and not in the left column
    {
        tempHaloBotRow0Next[0] = tempTileBotRow0Next[0] = computePoint(
            tempBotRow4Current[0], tempBotRow3Current[0], tempBotRow1Current[0], tempBotRow0Current[0],
            tempWestHaloZoneCurrent[_edgeSizes.localHeight - 1].first, tempWestHaloZoneCurrent[_edgeSizes.localHeight - 1].second, tempBotRow2Current[1], tempBotRow2Current[2],
            tempBotRow2Current[0],
            domainParamsBotRow4[0], domainParamsBotRow3[0], domainParamsBotRow1[0], domainParamsBotRow0[0],
            domainParamsWestHaloZone[_edgeSizes.localHeight - 1].first, domainParamsWestHaloZone[_edgeSizes.localHeight - 1].second, domainParamsBotRow2[1], domainParamsBotRow2[2],
            domainParamsBotRow2[0],
            domainMapBotCenter0[0]
        );
        tempHaloBotRow0Next[1] = tempTileBotRow0Next[1] = computePoint(
            tempBotRow4Current[1], tempBotRow3Current[1], tempBotRow1Current[1], tempBotRow0Current[1],
            tempWestHaloZoneCurrent[_edgeSizes.localHeight - 1].second, tempBotRow2Current[0], tempBotRow2Current[2], tempBotRow2Current[3],
            tempBotRow2Current[1],
            domainParamsBotRow4[1], domainParamsBotRow3[1], domainParamsBotRow1[1], domainParamsBotRow0[1],
            domainParamsWestHaloZone[_edgeSizes.localHeight - 1].second, domainParamsBotRow2[0], domainParamsBotRow2[2], domainParamsBotRow2[3],
            domainParamsBotRow2[1],
            domainMapBotCenter0[1]
        );

        tempHaloBotRow1Next[0] = tempTileBotRow1Next[0] = computePoint(
            tempBotRow5Current[0], tempBotRow4Current[0], tempBotRow2Current[0], tempBotRow1Current[0],
            tempWestHaloZoneCurrent[_edgeSizes.localHeight - 2].first, tempWestHaloZoneCurrent[_edgeSizes.localHeight - 2].second, tempBotRow3Current[1], tempBotRow3Current[2],
            tempBotRow3Current[0],
            domainParamsBotRow5[0], domainParamsBotRow4[0], domainParamsBotRow2[0], domainParamsBotRow1[0],
            domainParamsWestHaloZone[_edgeSizes.localHeight - 2].first, domainParamsWestHaloZone[_edgeSizes.localHeight - 2].second, domainParamsBotRow3[1], domainParamsBotRow3[2],
            domainParamsBotRow3[0],
            domainMapBotCenter1[0]
        );
        tempHaloBotRow1Next[1] = tempTileBotRow1Next[1] = computePoint(
            tempBotRow5Current[1], tempBotRow4Current[1], tempBotRow2Current[1], tempBotRow1Current[1],
            tempWestHaloZoneCurrent[_edgeSizes.localHeight - 2].second, tempBotRow3Current[0], tempBotRow3Current[2], tempBotRow3Current[3],
            tempBotRow3Current[1],
            domainParamsBotRow5[1], domainParamsBotRow4[1], domainParamsBotRow2[1], domainParamsBotRow1[1],
            domainParamsWestHaloZone[_edgeSizes.localHeight - 2].second, domainParamsBotRow3[0], domainParamsBotRow3[2], domainParamsBotRow3[3],
            domainParamsBotRow3[1],
            domainMapBotCenter1[1]
        );
    }

    // bottom row
    if (!isBottomRow()) // node is not in the bottom row
    {
        for (size_t i = 2; i < _edgeSizes.localWidth - 2; i++)
        {
            tempHaloBotRow0Next[i] = tempTileBotRow0Next[i] = computePoint(
                tempBotRow4Current[i], tempBotRow3Current[i], tempBotRow1Current[i], tempBotRow0Current[i],
                tempBotRow2Current[i - 2], tempBotRow2Current[i - 1], tempBotRow2Current[i + 1], tempBotRow2Current[i + 2],
                tempBotRow2Current[i],
                domainParamsBotRow4[i], domainParamsBotRow3[i], domainParamsBotRow1[i], domainParamsBotRow0[i],
                domainParamsBotRow2[i - 2], domainParamsBotRow2[i - 1], domainParamsBotRow2[i + 1], domainParamsBotRow2[i + 2],
                domainParamsBotRow2[i],
                domainMapTile[i]
            );
            tempHaloBotRow1Next[i] = tempTileBotRow1Next[i] = computePoint(
                tempBotRow5Current[i], tempBotRow4Current[i], tempBotRow2Current[i], tempBotRow1Current[i],
                tempBotRow3Current[i - 2], tempBotRow3Current[i - 1], tempBotRow3Current[i + 1], tempBotRow3Current[i + 2],
                tempBotRow3Current[i],
                domainParamsBotRow5[i], domainParamsBotRow4[i], domainParamsBotRow2[i], domainParamsBotRow1[i],
                domainParamsBotRow3[i - 2], domainParamsBotRow3[i - 1], domainParamsBotRow3[i + 1], domainParamsBotRow3[i + 2],
                domainParamsBotRow3[i],
                domainMapTile[i]
            );
        }
    }

    // right lower corner
    if (!isBottomRow() && !isRightColumn()) // node is not in the bottom row and not in the right column
    {
        tempHaloBotRow0Next[_edgeSizes.localWidth - 2] = tempTileBotRow0Next[_edgeSizes.localWidth - 2] = computePoint(
            tempBotRow4Current[_edgeSizes.localWidth - 2], tempBotRow3Current[_edgeSizes.localWidth - 2], tempBotRow1Current[_edgeSizes.localWidth - 2], tempBotRow0Current[_edgeSizes.localWidth - 2],
            tempBotRow2Current[_edgeSizes.localWidth - 4], tempBotRow2Current[_edgeSizes.localWidth - 3], tempBotRow2Current[_edgeSizes.localWidth - 1], tempEastHaloZoneCurrent[_edgeSizes.localHeight - 1].first,
            tempBotRow2Current[_edgeSizes.localWidth - 2],
            domainParamsBotRow4[_edgeSizes.localWidth - 2], domainParamsBotRow3[_edgeSizes.localWidth - 2], domainParamsBotRow1[_edgeSizes.localWidth - 2], domainParamsBotRow0[_edgeSizes.localWidth - 2],
            domainParamsBotRow2[_edgeSizes.localWidth - 4], domainParamsBotRow2[_edgeSizes.localWidth - 3], domainParamsBotRow2[_edgeSizes.localWidth - 1], domainParamsEastHaloZone[_edgeSizes.localHeight - 1].first,
            domainParamsBotRow2[_edgeSizes.localWidth - 2],
            domainMapBotCenter0[_edgeSizes.localWidth - 2]
        );
        tempHaloBotRow0Next[_edgeSizes.localWidth - 1] = tempTileBotRow0Next[_edgeSizes.localWidth - 1] = computePoint(
            tempBotRow4Current[_edgeSizes.localWidth - 1], tempBotRow3Current[_edgeSizes.localWidth - 1], tempBotRow1Current[_edgeSizes.localWidth - 1], tempBotRow0Current[_edgeSizes.localWidth - 1],
            tempBotRow2Current[_edgeSizes.localWidth - 3], tempBotRow2Current[_edgeSizes.localWidth - 2], tempEastHaloZoneCurrent[_edgeSizes.localHeight - 1].first, tempEastHaloZoneCurrent[_edgeSizes.localHeight - 1].second,
            tempBotRow2Current[_edgeSizes.localWidth - 1],
            domainParamsBotRow4[_edgeSizes.localWidth - 1], domainParamsBotRow3[_edgeSizes.localWidth - 1], domainParamsBotRow1[_edgeSizes.localWidth - 1], domainParamsBotRow0[_edgeSizes.localWidth - 1],
            domainParamsBotRow2[_edgeSizes.localWidth - 3], domainParamsBotRow2[_edgeSizes.localWidth - 2], domainParamsEastHaloZone[_edgeSizes.localHeight - 1].first, domainParamsEastHaloZone[_edgeSizes.localHeight - 1].second,
            domainParamsBotRow2[_edgeSizes.localWidth - 1],
            domainMapBotCenter0[_edgeSizes.localWidth - 1]
        );

        tempHaloBotRow1Next[_edgeSizes.localWidth - 2] = tempTileBotRow1Next[_edgeSizes.localWidth - 2] = computePoint(
            tempBotRow5Current[_edgeSizes.localWidth - 2], tempBotRow4Current[_edgeSizes.localWidth - 2], tempBotRow2Current[_edgeSizes.localWidth - 2], tempBotRow1Current[_edgeSizes.localWidth - 2],
            tempBotRow3Current[_edgeSizes.localWidth - 4], tempBotRow3Current[_edgeSizes.localWidth - 3], tempBotRow3Current[_edgeSizes.localWidth - 1], tempEastHaloZoneCurrent[_edgeSizes.localHeight - 2].first,
            tempBotRow3Current[_edgeSizes.localWidth - 2],
            domainParamsBotRow5[_edgeSizes.localWidth - 2], domainParamsBotRow4[_edgeSizes.localWidth - 2], domainParamsBotRow2[_edgeSizes.localWidth - 2], domainParamsBotRow1[_edgeSizes.localWidth - 2],
            domainParamsBotRow3[_edgeSizes.localWidth - 4], domainParamsBotRow3[_edgeSizes.localWidth - 3], domainParamsBotRow3[_edgeSizes.localWidth - 1], domainParamsEastHaloZone[_edgeSizes.localHeight - 2].first,
            domainParamsBotRow3[_edgeSizes.localWidth - 2],
            domainMapBotCenter1[_edgeSizes.localWidth - 2]
        );
        tempHaloBotRow1Next[_edgeSizes.localWidth - 1] = tempTileBotRow1Next[_edgeSizes.localWidth - 1] = computePoint(
            tempBotRow5Current[_edgeSizes.localWidth - 1], tempBotRow4Current[_edgeSizes.localWidth - 1], tempBotRow2Current[_edgeSizes.localWidth - 1], tempBotRow1Current[_edgeSizes.localWidth - 1],
            tempBotRow3Current[_edgeSizes.localWidth - 3], tempBotRow3Current[_edgeSizes.localWidth - 2], tempEastHaloZoneCurrent[_edgeSizes.localHeight - 2].first, tempEastHaloZoneCurrent[_edgeSizes.localHeight - 2].second,
            tempBotRow3Current[_edgeSizes.localWidth - 1],
            domainParamsBotRow5[_edgeSizes.localWidth - 1], domainParamsBotRow4[_edgeSizes.localWidth - 1], domainParamsBotRow2[_edgeSizes.localWidth - 1], domainParamsBotRow1[_edgeSizes.localWidth - 1],
            domainParamsBotRow3[_edgeSizes.localWidth - 3], domainParamsBotRow3[_edgeSizes.localWidth - 2], domainParamsEastHaloZone[_edgeSizes.localHeight - 2].first, domainParamsEastHaloZone[_edgeSizes.localHeight - 2].second,
            domainParamsBotRow3[_edgeSizes.localWidth - 1],
            domainMapBotCenter1[_edgeSizes.localWidth - 1]
        );
    }
}

void ParallelHeatSolver::computeTiles(bool current, bool next)
{
    float *tempCurrentTile = _tempTiles[current].data();
    float *tempNextTile = _tempTiles[next].data();

    for (size_t i = 2; i < _edgeSizes.localHeight - 2; i++)
    {
        for (size_t j = 2; j < _edgeSizes.localWidth - 2; j++)
        {
            tempNextTile[i * _edgeSizes.localWidth + j] = computePoint(
                tempCurrentTile[(i - 2) * _edgeSizes.localWidth + j], tempCurrentTile[(i - 1) * _edgeSizes.localWidth + j], tempCurrentTile[(i + 1) * _edgeSizes.localWidth + j], tempCurrentTile[(i + 2) * _edgeSizes.localWidth + j], 
                tempCurrentTile[i * _edgeSizes.localWidth + j - 2], tempCurrentTile[i * _edgeSizes.localWidth + j - 1], tempCurrentTile[i * _edgeSizes.localWidth + j + 1], tempCurrentTile[i * _edgeSizes.localWidth + j + 2],
                tempCurrentTile[i * _edgeSizes.localWidth + j],
                _domainParamsTile[(i - 2) * _edgeSizes.localWidth + j], _domainParamsTile[(i - 1) * _edgeSizes.localWidth + j], _domainParamsTile[(i + 1) * _edgeSizes.localWidth + j], _domainParamsTile[(i + 2) * _edgeSizes.localWidth + j],
                _domainParamsTile[i * _edgeSizes.localWidth + j - 2], _domainParamsTile[i * _edgeSizes.localWidth + j - 1], _domainParamsTile[i * _edgeSizes.localWidth + j + 1], _domainParamsTile[i * _edgeSizes.localWidth + j + 2],
                _domainParamsTile[i * _edgeSizes.localWidth + j],
                _domainMapTile[i * _edgeSizes.localWidth + j]
            );
        }
    }
}

void ParallelHeatSolver::startHaloExchangeP2P(bool current, bool next)
{
    /**********************************************************************************************************************/
    /*                       Start the non-blocking halo zones exchange using P2P communication.                          */
    /*                         Use the requests array to return the requests from the function.                           */
    /*                            Don't forget to set the empty requests to MPI_REQUEST_NULL.                             */
    /**********************************************************************************************************************/
    MPI_Ineighbor_alltoallv(_tempHaloZones[next].data(), _transferCounts, _displacements, MPI_FLOAT,
                            _tempHaloZones[current].data(), _transferCounts, _displacements, MPI_FLOAT, _topologyComm, &_haloExchangeRequest);
}

void ParallelHeatSolver::startHaloExchangeRMA(float *localData, MPI_Win window)
{
    /**********************************************************************************************************************/
    /*                       Start the non-blocking halo zones exchange using RMA communication.                          */
    /*                   Do not forget that you put/get the values to/from the target's opposite side                     */
    /**********************************************************************************************************************/
}

void ParallelHeatSolver::awaitHaloExchangeP2P()
{
    /**********************************************************************************************************************/
    /*                       Wait for all halo zone exchanges to finalize using P2P communication.                        */
    /**********************************************************************************************************************/
    MPI_Status status = {0, 0, 0, 0, 0};
    MPI_Wait(&_haloExchangeRequest, &status);
    if (status.MPI_ERROR != MPI_SUCCESS)
    {
        cerr << "Rank: " << _worldRank << " - Error in halo exchange: " << status.MPI_ERROR << endl;
        MPI_Abort(MPI_COMM_WORLD, status.MPI_ERROR);
    }

    //int actualRecvCount;
    //MPI_Get_count(&status, MPI_FLOAT, &actualRecvCount);
    //if (actualRecvCount != _offsets.northSouthHalo * 2 + _offsets.westEastHalo * 2)
    //{
    //    cerr << "Rank: " << _worldRank << " - Error in halo exchange: received unexpected number of values. Expected: " <<  _offsets.northSouthHalo * 2 + _offsets.westEastHalo * 2  << " Received: " << actualRecvCount << endl;
    //    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COUNT);
    //}
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

    copy(_domainParamsTile.begin(), _domainParamsTile.begin() + _offsets.northSouthHalo, _domainParamsHaloZoneTmp.begin()); // North
    copy(_domainParamsTile.end() - _offsets.northSouthHalo, _domainParamsTile.end(), _domainParamsHaloZoneTmp.begin() + _offsets.northSouthHalo); // South

    // copy to halo zones West and East
    for (size_t i = 0; i < _edgeSizes.localHeight; i++)
    {
        // use the odd halo zone, so that after the initial exchange, the even halo zone can be used for the first iteration
        _tempHaloZones[1][2 * _offsets.northSouthHalo + 2 * i] = _tempTiles[0][i * _edgeSizes.localWidth]; // West
        _tempHaloZones[1][2 * _offsets.northSouthHalo + 2 * i + 1] = _tempTiles[0][i * _edgeSizes.localWidth + 1]; // West
        _tempHaloZones[1][2 * _offsets.northSouthHalo + _offsets.westEastHalo + 2 * i] = _tempTiles[0][i * _edgeSizes.localWidth + _edgeSizes.localWidth - 2]; // East
        _tempHaloZones[1][2 * _offsets.northSouthHalo + _offsets.westEastHalo + 2 * i + 1] = _tempTiles[0][i * _edgeSizes.localWidth + _edgeSizes.localWidth - 1]; // East

        _domainParamsHaloZoneTmp[2 * _offsets.northSouthHalo + 2 * i] = _domainParamsTile[i * _edgeSizes.localWidth]; // West
        _domainParamsHaloZoneTmp[2 * _offsets.northSouthHalo + 2 * i + 1] = _domainParamsTile[i * _edgeSizes.localWidth + 1]; // West
        _domainParamsHaloZoneTmp[2 * _offsets.northSouthHalo + _offsets.westEastHalo + 2 * i] = _domainParamsTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 2]; // East
        _domainParamsHaloZoneTmp[2 * _offsets.northSouthHalo + _offsets.westEastHalo + 2 * i + 1] = _domainParamsTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 1]; // East
    }

    _transferCounts[0] = _transferCounts[1] = _offsets.northSouthHalo;
    _transferCounts[2] = _transferCounts[3] = _offsets.westEastHalo;
    _displacements[0] = 0;
    _displacements[1] = _offsets.northSouthHalo;
    _displacements[2] = 2 * _offsets.northSouthHalo;
    _displacements[3] = 2 * _offsets.northSouthHalo + _offsets.westEastHalo;
}

void ParallelHeatSolver::exchangeInitialHaloZones()
{
    // scatter/gather the halo zones across neighbors 
    MPI_Neighbor_alltoallv(_tempHaloZones[1].data(), _transferCounts, _displacements, MPI_FLOAT,
                           _tempHaloZones[0].data(), _transferCounts, _displacements, MPI_FLOAT, _topologyComm);
    MPI_Neighbor_alltoallv(_domainParamsHaloZoneTmp.data(), _transferCounts, _displacements, MPI_FLOAT,
                           _domainParamsHaloZone.data(), _transferCounts, _displacements, MPI_FLOAT, _topologyComm);
}

void ParallelHeatSolver::run(std::vector<float, AlignedAllocator<float>> &outResult)
{
    #if MANIPULATE_TEMP
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

    #if (PRINT_DEBUG && false)
        if (_worldRank == 0)
        {
            cerr << "Initial domain params: " << endl;
            for (int i = 0; i < _edgeSizes.global; i++)
            {
                for (int j = 0; j < _edgeSizes.global; j++)
                {
                    cerr << mMaterialProps.getDomainParameters()[i * _edgeSizes.global + j] << " ";
                }
                cerr << endl;
            }
        }
    #endif

    scatterInitialData();
    prepareInitialHaloZones();
    exchangeInitialHaloZones();   
    
    #if (PRINT_DEBUG && 0)
        for (int i = 0; i < _worldSize; i++)
        {
            printHalo(i, 0);
        }
    #endif

    // copy initial temperature to the second buffer
    copy(_tempTiles[0].begin(), _tempTiles[0].end(), _tempTiles[1].begin());

    // deallocate the temporary buffers
    _domainParamsHaloZoneTmp.resize(0);
    _initialScatterTemp.resize(0);
    _initialScatterDomainParams.resize(0);
    _initialScatterDomainMap.resize(0);

    double startTime = MPI_Wtime();

    // 3. Start main iterative simulation loop.
    for (std::size_t iter = 0; iter < mSimulationProps.getNumIterations(); ++iter)
    {
        const bool current = iter & 1;
        const bool next = iter ^ 1;

        /**********************************************************************************************************************/
        /*                            Compute and exchange halo zones using P2P or RMA.                                       */
        /**********************************************************************************************************************/
        computeHaloZones(current, next);
        startHaloExchangeP2P(current, next);
        computeTiles(current, next);
        awaitHaloExchangeP2P();

        if (iter == 2)
        {
            for (int i = 0; i < _worldSize; i++)
            {
                printTile(i, next);
            }
            return;
        }

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
