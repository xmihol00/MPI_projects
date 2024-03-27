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
    : HeatSolverBase(simulationProps, materialProps), 
      _simulationHyperParams{.airFlowRate = mSimulationProps.getAirflowRate(),
                             .coolerTemp = mMaterialProps.getCoolerTemperature()}
{
    MPI_Comm_size(MPI_COMM_WORLD, &_worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &_worldRank);

    mSimulationProps.getDecompGrid(_decomposition.nx, _decomposition.ny);

    _sizes.globalEdge = mMaterialProps.getEdgeSize();
    _sizes.localWidth = _sizes.globalEdge / _decomposition.nx;
    _sizes.localHeight = _sizes.globalEdge / _decomposition.ny;
    _sizes.localHeightWithHalos = _sizes.localHeight + 4;
    _sizes.localWidthWithHalos = _sizes.localWidth + 4;
    _sizes.localTile = _sizes.localHeight * _sizes.localWidth;
    _sizes.localTileWithHalos = _sizes.localWidthWithHalos * _sizes.localHeightWithHalos;
    _sizes.globalTile = _sizes.globalEdge * _sizes.globalEdge;
    _sizes.northSouthHalo = 2 * _sizes.localWidth;
    _sizes.westEastHalo = 2 * _sizes.localHeight;

    /**********************************************************************************************************************/
    /*                                  Call init* and alloc* methods in correct order                                    */
    /**********************************************************************************************************************/

    allocLocalTiles();
    initGridTopology();
    initDataTypes();

    if (!mSimulationProps.getOutputFileName().empty())
    {
        /**********************************************************************************************************************/
        /*                               Open output file if output file name was specified.                                  */
        /*  If mSimulationProps.useParallelIO() flag is set to true, open output file for parallel access, otherwise open it  */
        /*                         only on MASTER rank using sequetial IO. Use openOutputFile* methods.                       */
        /**********************************************************************************************************************/
        if (mSimulationProps.useParallelIO())
        {
            openOutputFileParallel();
        }
        else
        {
            if (_worldRank == 0)
            {
                openOutputFileSequential();
            }
        }
    }
}

ParallelHeatSolver::~ParallelHeatSolver()
{
    /**********************************************************************************************************************/
    /*                                  Call deinit* and dealloc* methods in correct order                                */
    /*                                             (should be in reverse order)                                           */
    /**********************************************************************************************************************/

    deinitDataTypes();
    deinitGridTopology();
}

string_view ParallelHeatSolver::getCodeType() const
{
    return codeType;
}

void ParallelHeatSolver::initGridTopology()
{
    // 2D grid topology without periodicity
    MPI_Cart_create(MPI_COMM_WORLD, 2, array<int, 2>{_decomposition.ny, _decomposition.nx}.data(), array<int, 2>{false, false}.data(), 0, &_topologyComm);
    MPI_Comm_set_name(_topologyComm, "Topology Communicator");

    // there is always an even number of columns, therefore there are two middle columns (the computation is symmetric),
    // in this case the right middle colum is used (left most column in tiles of the right middle column of nodes)
    int middleColModulo = _decomposition.nx >> 1;
    if (_worldRank % _decomposition.nx == middleColModulo) // right middle column nodes
    {
        MPI_Comm_split(MPI_COMM_WORLD, 0, _worldRank / _decomposition.nx, &_midColComm);
        MPI_Comm_set_name(_midColComm, "Middle Column Communicator");
        MPI_Comm_rank(_midColComm, &_midColRank);
    }
    else // other columns
    {
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, MPI_UNDEFINED, &_midColComm);
        _midColComm = MPI_COMM_NULL;
    }

    if (mSimulationProps.isRunParallelRMA())
    {
    #if DATA_TYPE_EXCHANGE
        MPI_Win_create(_tempTiles[0].data(), _tempTiles[0].size() * sizeof(float), sizeof(float), MPI_INFO_NULL, _topologyComm, &_haloExchangeWindows[0]);
        MPI_Win_create(_tempTiles[1].data(), _tempTiles[1].size() * sizeof(float), sizeof(float), MPI_INFO_NULL, _topologyComm, &_haloExchangeWindows[1]);
    #elif RAW_EXCHANGE
        MPI_Win_create(_tempHaloZones[0].data(), _tempHaloZones[0].size() * sizeof(float), sizeof(float), MPI_INFO_NULL, _topologyComm, &_haloExchangeWindows[0]);
    #endif
    }
    
    // get the neighbors of the current node
    MPI_Cart_shift(_topologyComm, 0, 1, _neighbors, _neighbors + 1);
    MPI_Cart_shift(_topologyComm, 1, 1, _neighbors + 2, _neighbors + 3);

    // counts and displacements for all to all communication with different data sizes
    _transferCounts[0] = _transferCounts[1] = _sizes.northSouthHalo;
    _transferCounts[2] = _transferCounts[3] = _sizes.westEastHalo;
    _displacements[0] = 0;
    _displacements[1] = _sizes.northSouthHalo;
    _displacements[2] = 2 * _sizes.northSouthHalo;
    _displacements[3] = 2 * _sizes.northSouthHalo + _sizes.westEastHalo;

    // necessary for RMA communication, where when putting data e.g. to the north neighbor, the data must be put to its south halo zone
    _inverseDisplacements[0] = _sizes.northSouthHalo;
    _inverseDisplacements[1] = 0;
    _inverseDisplacements[2] = 2 * _sizes.northSouthHalo + _sizes.westEastHalo;
    _inverseDisplacements[3] = 2 * _sizes.northSouthHalo;
}

void ParallelHeatSolver::deinitGridTopology()
{
    MPI_Comm_free(&_topologyComm);
    if (_midColComm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&_midColComm);
    }

    if (mSimulationProps.isRunParallelRMA())
    {
    #if DATA_TYPE_EXCHANGE
        MPI_Win_free(&_haloExchangeWindows[0]);
        MPI_Win_free(&_haloExchangeWindows[1]);
    #elif RAW_EXCHANGE
        MPI_Win_free(&_haloExchangeWindows[0]);
    #endif
    }
}

void ParallelHeatSolver::allocLocalTiles()
{
#if DATA_TYPE_EXCHANGE
    _tempTiles[0].resize(_sizes.localTileWithHalos);
    _tempTiles[1].resize(_sizes.localTileWithHalos);
    _domainParamsTile.resize(_sizes.localTileWithHalos);
    _domainMapTile.resize(_sizes.localTileWithHalos);
#elif RAW_EXCHANGE
    _tempTiles[0].resize(_sizes.localTile);
    _tempTiles[1].resize(_sizes.localTile);
    _domainParamsTile.resize(_sizes.localTile);
    _domainMapTile.resize(_sizes.localTile);

    _tempHaloZones[0].resize(_sizes.localWidth * 4 + _sizes.localHeight * 4);
    _tempHaloZones[1].resize(_sizes.localWidth * 4 + _sizes.localHeight * 4);
    _domainParamsHaloZoneTmp.resize(_sizes.localWidth * 4 + _sizes.localHeight * 4);
    _domainParamsHaloZone.resize(_sizes.localWidth * 4 + _sizes.localHeight * 4);
#endif
}

void ParallelHeatSolver::initDataTypes()
{
    int tileSize[2] = {_sizes.globalEdge, _sizes.globalEdge};
    int localTileSize[2] = {_sizes.localHeight, _sizes.localWidth};
    int starts[2] = {0, 0};
    
    // temperature/domain parameters scatter and gather data type
    MPI_Type_create_subarray(2, tileSize, localTileSize, starts, MPI_ORDER_C, MPI_FLOAT, &_floatTileWithoutHaloZones);
    MPI_Type_commit(&_floatTileWithoutHaloZones);
    MPI_Type_create_resized(_floatTileWithoutHaloZones, 0, _sizes.localWidth * sizeof(float), &_floatTileWithoutHaloZonesResized);
    MPI_Type_commit(&_floatTileWithoutHaloZonesResized);

    // domain map scatter data type
    MPI_Type_create_subarray(2, tileSize, localTileSize, starts, MPI_ORDER_C, MPI_INT, &_intTileWithoutHaloZones);
    MPI_Type_commit(&_intTileWithoutHaloZones);
    MPI_Type_create_resized(_intTileWithoutHaloZones, 0, _sizes.localWidth * sizeof(int), &_intTileWithoutHaloZonesResized);
    MPI_Type_commit(&_intTileWithoutHaloZonesResized);

    tileSize[0] = _sizes.localHeight + 4;
    tileSize[1] = _sizes.localWidth + 4;
    starts[0] = 2;
    starts[1] = 2;

    // temperature/ domain parameters local tile with halo zones data type
    MPI_Type_create_subarray(2, tileSize, localTileSize, starts, MPI_ORDER_C, MPI_FLOAT, &_floatTileWithHaloZones);
    MPI_Type_commit(&_floatTileWithHaloZones);

    // domain map local tile with halo zones data type
    MPI_Type_create_subarray(2, tileSize, localTileSize, starts, MPI_ORDER_C, MPI_INT, &_intTileWithHaloZones);
    MPI_Type_commit(&_intTileWithHaloZones);

    _scatterGatherCounts.resize(_worldSize);
    _scatterGatherDisplacements.resize(_worldSize);
    for (int i = 0; i < _worldSize; i++)
    {
        _scatterGatherCounts[i] = 1;
        // ensure displacements are increasing by 1 on the same row of nodes and between rows of nodes by the multiple of local tile hight and width
        // i.e. each row of nodes must displace itself by multiple of whole tiles, while within the row of nodes the displacement is by 1
        _scatterGatherDisplacements[i] = (i % _decomposition.nx) + _sizes.localHeight * (i / _decomposition.nx) * _decomposition.nx;
    }

    localTileSize[0] = 2;
    localTileSize[1] = _sizes.localWidth;

    // temperature/domain parameters north halo zones for all to all exchange
    starts[0] = 2;
    MPI_Type_create_subarray(2, tileSize, localTileSize, starts, MPI_ORDER_C, MPI_FLOAT, &_floatSendHaloZoneTypes[NORTH]);
    MPI_Type_commit(&_floatSendHaloZoneTypes[NORTH]);
    starts[0] = 0;
    MPI_Type_create_subarray(2, tileSize, localTileSize, starts, MPI_ORDER_C, MPI_FLOAT, &_floatRecvHaloZoneTypes[NORTH]);
    MPI_Type_commit(&_floatRecvHaloZoneTypes[NORTH]);

    // temperature/domain parameters south halo zones for all to all exchange
    starts[0] = _sizes.localHeight;
    MPI_Type_create_subarray(2, tileSize, localTileSize, starts, MPI_ORDER_C, MPI_FLOAT, &_floatSendHaloZoneTypes[SOUTH]);
    MPI_Type_commit(&_floatSendHaloZoneTypes[SOUTH]);
    starts[0] = _sizes.localHeight + 2;
    MPI_Type_create_subarray(2, tileSize, localTileSize, starts, MPI_ORDER_C, MPI_FLOAT, &_floatRecvHaloZoneTypes[SOUTH]);
    MPI_Type_commit(&_floatRecvHaloZoneTypes[SOUTH]);

    localTileSize[0] = _sizes.localHeight;
    localTileSize[1] = 2;

    // temperature/domain parameters west halo zones for all to all exchange
    starts[0] = 2;
    MPI_Type_create_subarray(2, tileSize, localTileSize, starts, MPI_ORDER_C, MPI_FLOAT, &_floatSendHaloZoneTypes[WEST]);
    MPI_Type_commit(&_floatSendHaloZoneTypes[WEST]);
    starts[1] = 0;
    MPI_Type_create_subarray(2, tileSize, localTileSize, starts, MPI_ORDER_C, MPI_FLOAT, &_floatRecvHaloZoneTypes[WEST]);
    MPI_Type_commit(&_floatRecvHaloZoneTypes[WEST]);

    // temperature/domain parameters east halo zones for all to all exchange
    starts[1] = _sizes.localWidth;
    MPI_Type_create_subarray(2, tileSize, localTileSize, starts, MPI_ORDER_C, MPI_FLOAT, &_floatSendHaloZoneTypes[EAST]);
    MPI_Type_commit(&_floatSendHaloZoneTypes[EAST]);
    starts[1] = _sizes.localWidth + 2;
    MPI_Type_create_subarray(2, tileSize, localTileSize, starts, MPI_ORDER_C, MPI_FLOAT, &_floatRecvHaloZoneTypes[EAST]);
    MPI_Type_commit(&_floatRecvHaloZoneTypes[EAST]);

    _floatInverseRecvHaloZoneTypes[NORTH] = _floatRecvHaloZoneTypes[SOUTH];
    _floatInverseRecvHaloZoneTypes[SOUTH] = _floatRecvHaloZoneTypes[NORTH];
    _floatInverseRecvHaloZoneTypes[WEST] = _floatRecvHaloZoneTypes[EAST];
    _floatInverseRecvHaloZoneTypes[EAST] = _floatRecvHaloZoneTypes[WEST];
}

void ParallelHeatSolver::deinitDataTypes()
{
    MPI_Type_free(&_floatTileWithoutHaloZones);
    MPI_Type_free(&_floatTileWithoutHaloZonesResized);
    MPI_Type_free(&_intTileWithoutHaloZones);
    MPI_Type_free(&_intTileWithoutHaloZonesResized);
    MPI_Type_free(&_floatTileWithHaloZones);
    MPI_Type_free(&_intTileWithHaloZones);

    MPI_Type_free(&_floatSendHaloZoneTypes[NORTH]);
    MPI_Type_free(&_floatSendHaloZoneTypes[SOUTH]);
    MPI_Type_free(&_floatSendHaloZoneTypes[WEST]);
    MPI_Type_free(&_floatSendHaloZoneTypes[EAST]);
    MPI_Type_free(&_floatRecvHaloZoneTypes[SOUTH]);
    MPI_Type_free(&_floatRecvHaloZoneTypes[NORTH]);
    MPI_Type_free(&_floatRecvHaloZoneTypes[WEST]);
    MPI_Type_free(&_floatRecvHaloZoneTypes[EAST]);
}

void ParallelHeatSolver::computeTempHaloZones_Raw(bool current, bool next)
{
    enum class Corners { LEFT_UPPER, RIGHT_UPPER, LEFT_LOWER, RIGHT_LOWER };
    enum class CornerPoints { LEFT_UPPER, RIGHT_UPPER, LEFT_LOWER, RIGHT_LOWER };
    auto computeCornerPoint = [&]<Corners corner, CornerPoints cornerPoint>() 
    {   
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
                tempCenter = _tempTiles[current][rowOffset];
                tempSouth = _tempTiles[current][_sizes.localWidth + rowOffset];
                tempSouthLower = _tempTiles[current][2 * _sizes.localWidth + rowOffset];

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
                tempNorth = _tempTiles[current][rowOffset - _sizes.localWidth];
                tempCenter = _tempTiles[current][rowOffset];
                tempSouth = _tempTiles[current][_sizes.localWidth + rowOffset];
                tempSouthLower = _tempTiles[current][2 * _sizes.localWidth + rowOffset];

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
                tempNorth = _tempTiles[current][rowOffset - 2 * _sizes.localWidth];
                tempNorthUpper = _tempTiles[current][rowOffset - _sizes.localWidth];
                tempCenter = _tempTiles[current][rowOffset];
                tempSouth = _tempTiles[current][rowOffset + _sizes.localWidth];
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
                tempNorth = _tempTiles[current][rowOffset - 2 * _sizes.localWidth];
                tempNorthUpper = _tempTiles[current][rowOffset - _sizes.localWidth];
                tempCenter = _tempTiles[current][rowOffset];
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
                tempEast = _tempTiles[current][rowOffset + 1];
                tempEastRight = _tempTiles[current][rowOffset + 2];

                domainParamsWestLeft = domainParamsWestHaloZone[haloZoneOffset].first;
                domainParamsWest = domainParamsWestHaloZone[haloZoneOffset].second;
                domainParamsEast = _domainParamsTile[rowOffset + 1];
                domainParamsEastRight = _domainParamsTile[rowOffset + 2];
            }
            else
            {
                tempWestLeft = tempWestHaloZone[haloZoneOffset].second;
                tempWest = _tempTiles[current][rowOffset - 1];
                tempEast = _tempTiles[current][rowOffset + 1];
                tempEastRight = _tempTiles[current][rowOffset + 2];

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
                tempWestLeft = _tempTiles[current][rowOffset - 2];
                tempWest = _tempTiles[current][rowOffset - 1];
                tempEast = _tempTiles[current][rowOffset + 1];
                tempEastRight = tempEastHaloZone[haloZoneOffset].first;

                domainParamsWestLeft = _domainParamsTile[rowOffset - 2];
                domainParamsWest = _domainParamsTile[rowOffset - 1];
                domainParamsEast = _domainParamsTile[rowOffset + 1];
                domainParamsEastRight = domainParamsEastHaloZone[haloZoneOffset].first;
            }
            else
            {
                tempWestLeft = _tempTiles[current][rowOffset - 2];
                tempWest = _tempTiles[current][rowOffset - 1];
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
    };

    // unpack data into separate pointers for easier access, simulate the complete tile with halo zones
    // temperature
    float *tempTopRow0Current = _tempHaloZones[0].data(); // closest to the top
    float *tempTopRow1Current = _tempHaloZones[0].data() + _sizes.localWidth;
    float *tempTopRow2Current = _tempTiles[current].data();
    float *tempTopRow3Current = _tempTiles[current].data() + _sizes.localWidth;
    float *tempTopRow4Current = _tempTiles[current].data() + _sizes.northSouthHalo;
    float *tempTopRow5Current = _tempTiles[current].data() + _sizes.northSouthHalo + _sizes.localWidth;

    float *tempBotRow5Current = _tempTiles[current].data() + _sizes.localTile - 2 * _sizes.northSouthHalo;
    float *tempBotRow4Current = _tempTiles[current].data() + _sizes.localTile - _sizes.northSouthHalo - _sizes.localWidth;
    float *tempBotRow3Current = _tempTiles[current].data() + _sizes.localTile - _sizes.northSouthHalo;
    float *tempBotRow2Current = _tempTiles[current].data() + _sizes.localTile - _sizes.localWidth;
    float *tempBotRow1Current = _tempHaloZones[0].data() + _sizes.northSouthHalo;
    float *tempBotRow0Current = _tempHaloZones[0].data() + _sizes.northSouthHalo + _sizes.localWidth; // closest to the bottom

    // cast to pairs for easier access
    pair<float, float> *tempWestHaloZoneCurrent = reinterpret_cast<pair<float, float> *>(_tempHaloZones[0].data() + 2 * _sizes.northSouthHalo);
    pair<float, float> *tempEastHaloZoneCurrent = reinterpret_cast<pair<float, float> *>(_tempHaloZones[0].data() + 2 * _sizes.northSouthHalo + _sizes.westEastHalo);

    // rows to store results to
    float *tempHaloTopRow0Next = _tempHaloZones[1].data(); // closest to the top
    float *tempHaloTopRow1Next = _tempHaloZones[1].data() + _sizes.localWidth;
    float *tempTileTopRow0Next = _tempTiles[next].data(); // closest to the top
    float *tempTileTopRow1Next = _tempTiles[next].data() + _sizes.localWidth;

    float *tempHaloBotRow1Next = _tempHaloZones[1].data() + _sizes.northSouthHalo;
    float *tempHaloBotRow0Next = _tempHaloZones[1].data() + _sizes.northSouthHalo + _sizes.localWidth; // closest to the bottom
    float *tempTileBotRow1Next = _tempTiles[next].data() + _sizes.localTile - _sizes.northSouthHalo;
    float *tempTileBotRow0Next = _tempTiles[next].data() + _sizes.localTile - _sizes.localWidth; // closest to the bottom

    // columns to store results to (only halo zones)
    pair<float, float> *tempWestHaloZoneNext = reinterpret_cast<pair<float, float> *>(_tempHaloZones[1].data() + 2 * _sizes.northSouthHalo);
    pair<float, float> *tempEastHaloZoneNext = reinterpret_cast<pair<float, float> *>(_tempHaloZones[1].data() + 2 * _sizes.northSouthHalo + _sizes.westEastHalo);

    float *tempCurrentTile = _tempTiles[current].data();
    float *tempNextTile = _tempTiles[next].data();

    // domain parameters
    float *domainParamsTopRow0 = _domainParamsHaloZone.data();
    float *domainParamsTopRow1 = _domainParamsHaloZone.data() + _sizes.localWidth;
    float *domainParamsTopRow2 = _domainParamsTile.data();
    float *domainParamsTopRow3 = _domainParamsTile.data() + _sizes.localWidth;
    float *domainParamsTopRow4 = _domainParamsTile.data() + _sizes.northSouthHalo;
    float *domainParamsTopRow5 = _domainParamsTile.data() + _sizes.northSouthHalo + _sizes.localWidth;

    float *domainParamsBotRow5 = _domainParamsTile.data() + _sizes.localTile - 2 * _sizes.northSouthHalo;
    float *domainParamsBotRow4 = _domainParamsTile.data() + _sizes.localTile - _sizes.northSouthHalo - _sizes.localWidth;
    float *domainParamsBotRow3 = _domainParamsTile.data() + _sizes.localTile - _sizes.northSouthHalo;
    float *domainParamsBotRow2 = _domainParamsTile.data() + _sizes.localTile - _sizes.localWidth;
    float *domainParamsBotRow1 = _domainParamsHaloZone.data() + _sizes.northSouthHalo;
    float *domainParamsBotRow0 = _domainParamsHaloZone.data() + _sizes.northSouthHalo + _sizes.localWidth;

    float *domainParamsTile = _domainParamsTile.data();

    pair<float, float> *domainParamsWestHaloZone = reinterpret_cast<pair<float, float> *>(_domainParamsHaloZone.data() + 2 * _sizes.northSouthHalo);
    pair<float, float> *domainParamsEastHaloZone = reinterpret_cast<pair<float, float> *>(_domainParamsHaloZone.data() + 2 * _sizes.northSouthHalo + _sizes.westEastHalo);

    // domain map
    int *domainMapTile = _domainMapTile.data();
    int *domainMapTopCenter0 = _domainMapTile.data();
    int *domainMapTopCenter1 = _domainMapTile.data() + _sizes.localWidth;

    int *domainMapBotCenter1 = _domainMapTile.data() + _sizes.localTile - _sizes.northSouthHalo;
    int *domainMapBotCenter0 = _domainMapTile.data() + _sizes.localTile - _sizes.localWidth;

    // left upper corner
    if (isNotTopRow() && isNotLeftColumn()) // node is not in the top row and not in the left column
    {
        // the corner consists of 4 values, which must be stored to the tile, north and west halo zones
        // most left most upper value
        tempWestHaloZoneNext[0].first = 
            tempHaloTopRow0Next[0] = 
            tempTileTopRow0Next[0] = computeCornerPoint.operator()<Corners::LEFT_UPPER, CornerPoints::LEFT_UPPER>();

        // second most left most upper value
        tempWestHaloZoneNext[0].second = 
            tempHaloTopRow0Next[1] = 
            tempTileTopRow0Next[1] = computeCornerPoint.operator()<Corners::LEFT_UPPER, CornerPoints::RIGHT_UPPER>();

        // most left second most upper value
        tempWestHaloZoneNext[1].first = 
            tempHaloTopRow1Next[0] = 
            tempTileTopRow1Next[0] = computeCornerPoint.operator()<Corners::LEFT_UPPER, CornerPoints::LEFT_LOWER>();

        // second most left second most upper value
        tempWestHaloZoneNext[1].second = 
            tempHaloTopRow1Next[1] = 
            tempTileTopRow1Next[1] = computeCornerPoint.operator()<Corners::LEFT_UPPER, CornerPoints::RIGHT_LOWER>();
    }

    // north halo zone
    if (isNotTopRow()) // node is not in the top row
    {
        for (size_t i = 2; i < _sizes.localWidth - 2; i++)
        {
            tempHaloTopRow0Next[i] = tempTileTopRow0Next[i] = computePoint(
                tempTopRow0Current[i], tempTopRow1Current[i], tempTopRow3Current[i], tempTopRow4Current[i],
                tempTopRow2Current[i - 2], tempTopRow2Current[i - 1], tempTopRow2Current[i + 1], tempTopRow2Current[i + 2],
                tempTopRow2Current[i],
                domainParamsTopRow0[i], domainParamsTopRow1[i], domainParamsTopRow3[i], domainParamsTopRow4[i],
                domainParamsTopRow2[i - 2], domainParamsTopRow2[i - 1], domainParamsTopRow2[i + 1], domainParamsTopRow2[i + 2],
                domainParamsTopRow2[i],
                domainMapTopCenter0[i]);
            tempHaloTopRow1Next[i] = tempTileTopRow1Next[i] = computePoint(
                tempTopRow1Current[i], tempTopRow2Current[i], tempTopRow4Current[i], tempTopRow5Current[i],
                tempTopRow3Current[i - 2], tempTopRow3Current[i - 1], tempTopRow3Current[i + 1], tempTopRow3Current[i + 2],
                tempTopRow3Current[i],
                domainParamsTopRow1[i], domainParamsTopRow2[i], domainParamsTopRow4[i], domainParamsTopRow5[i],
                domainParamsTopRow3[i - 2], domainParamsTopRow3[i - 1], domainParamsTopRow3[i + 1], domainParamsTopRow3[i + 2],
                domainParamsTopRow3[i],
                domainMapTopCenter1[i]);
        }
    }

    // right upper corner
    if (isNotTopRow() && isNotRightColumn()) // node is not in the top row and not in the right column
    {
        // the corner consists of 4 values, which must be stored to the tile, north and east halo zones
        int verticalIdx = _sizes.localWidth - 2;
        int haloZoneIdx = 0;

        // second most right most upper value
        tempEastHaloZoneNext[haloZoneIdx].first = 
            tempHaloTopRow0Next[verticalIdx] = 
            tempTileTopRow0Next[verticalIdx] = computeCornerPoint.operator()<Corners::RIGHT_UPPER, CornerPoints::LEFT_UPPER>();

        verticalIdx++;

        // most right most upper value
        tempEastHaloZoneNext[haloZoneIdx].second = 
            tempHaloTopRow0Next[verticalIdx] = 
            tempTileTopRow0Next[verticalIdx] = computeCornerPoint.operator()<Corners::RIGHT_UPPER, CornerPoints::RIGHT_UPPER>();

        verticalIdx--;
        haloZoneIdx++;

        // second most right second most upper value
        tempEastHaloZoneNext[haloZoneIdx].first = 
            tempHaloTopRow1Next[verticalIdx] = 
            tempTileTopRow1Next[verticalIdx] = computeCornerPoint.operator()<Corners::RIGHT_UPPER, CornerPoints::LEFT_LOWER>();

        verticalIdx++;

        // most right second most upper value
        tempEastHaloZoneNext[haloZoneIdx].second = 
            tempHaloTopRow1Next[verticalIdx] = 
            tempTileTopRow1Next[verticalIdx] = computeCornerPoint.operator()<Corners::RIGHT_UPPER, CornerPoints::RIGHT_LOWER>();
    }

    // west halo zone
    if (isNotLeftColumn()) // node is not in the left column
    {
        for (size_t i = 2; i < _sizes.localHeight - 2; i++)
        {
            int northUpperIdx = (i - 2) * _sizes.localWidth;
            int northIdx = (i - 1) * _sizes.localWidth;
            int centerIdx = i * _sizes.localWidth;
            int southIdx = (i + 1) * _sizes.localWidth;
            int southLowerIdx = (i + 2) * _sizes.localWidth;
            int eastIdx = i * _sizes.localWidth + 1;
            int eastRightIdx = i * _sizes.localWidth + 2;
            int westIdx = i * _sizes.localWidth;

            // left most column
            tempWestHaloZoneNext[i].first = tempNextTile[centerIdx] = computePoint(
                tempCurrentTile[northUpperIdx], tempCurrentTile[northIdx], tempCurrentTile[southIdx], tempCurrentTile[southLowerIdx],
                tempWestHaloZoneCurrent[i].first, tempWestHaloZoneCurrent[i].second, tempCurrentTile[eastIdx], tempCurrentTile[eastRightIdx],
                tempCurrentTile[centerIdx],
                domainParamsTile[northUpperIdx], domainParamsTile[northIdx], domainParamsTile[southIdx], domainParamsTile[southLowerIdx],
                domainParamsWestHaloZone[i].first, domainParamsWestHaloZone[i].second, domainParamsTile[eastIdx], domainParamsTile[eastRightIdx],
                domainParamsTile[centerIdx],
                domainMapTile[centerIdx]);

            northUpperIdx += 1;
            northIdx += 1;
            centerIdx += 1;
            southIdx += 1;
            southLowerIdx += 1;
            eastIdx += 1;
            eastRightIdx += 1;

            // second left most column
            tempWestHaloZoneNext[i].second = tempNextTile[centerIdx] = computePoint(
                tempCurrentTile[northUpperIdx], tempCurrentTile[northIdx], tempCurrentTile[southIdx], tempCurrentTile[southLowerIdx],
                tempWestHaloZoneCurrent[i].second, tempCurrentTile[westIdx], tempCurrentTile[eastIdx], tempCurrentTile[eastRightIdx],
                tempCurrentTile[centerIdx],
                domainParamsTile[northUpperIdx], domainParamsTile[northIdx], domainParamsTile[southIdx], domainParamsTile[southLowerIdx],
                domainParamsWestHaloZone[i].second, domainParamsTile[westIdx], domainParamsTile[eastIdx], domainParamsTile[eastRightIdx],
                domainParamsTile[centerIdx],
                domainMapTile[centerIdx]);
        }
    }

    // east halo zone
    if (isNotRightColumn()) // node is not in the right column
    {
        for (size_t i = 2; i < _sizes.localHeight - 2; i++)
        {
            int northUpperIdx = (i - 2) * _sizes.localWidth + _sizes.localWidth - 2;
            int northIdx = (i - 1) * _sizes.localWidth + _sizes.localWidth - 2;
            int centerIdx = i * _sizes.localWidth + _sizes.localWidth - 2;
            int southIdx = (i + 1) * _sizes.localWidth + _sizes.localWidth - 2;
            int southLowerIdx = (i + 2) * _sizes.localWidth + _sizes.localWidth - 2;
            int westLeftIdx = i * _sizes.localWidth + _sizes.localWidth - 4;
            int westIdx = i * _sizes.localWidth + _sizes.localWidth - 3;
            int eastIdx = i * _sizes.localWidth + _sizes.localWidth - 1;

            // second right most column
            tempEastHaloZoneNext[i].first = tempNextTile[centerIdx] = computePoint(
                tempCurrentTile[northUpperIdx], tempCurrentTile[northIdx], tempCurrentTile[southIdx], tempCurrentTile[southLowerIdx],
                tempCurrentTile[westLeftIdx], tempCurrentTile[westIdx], tempCurrentTile[eastIdx], tempEastHaloZoneCurrent[i].first,
                tempCurrentTile[centerIdx],
                domainParamsTile[northUpperIdx], domainParamsTile[northIdx], domainParamsTile[southIdx], domainParamsTile[southLowerIdx],
                domainParamsTile[westLeftIdx], domainParamsTile[westIdx], domainParamsTile[eastIdx], domainParamsEastHaloZone[i].first,
                domainParamsTile[centerIdx],
                domainMapTile[centerIdx]);

            northUpperIdx += 1;
            northIdx += 1;
            centerIdx += 1;
            southIdx += 1;
            southLowerIdx += 1;
            westLeftIdx += 1;
            westIdx += 1;

            // right most column
            tempEastHaloZoneNext[i].second = tempNextTile[centerIdx] = computePoint(
                tempCurrentTile[northUpperIdx], tempCurrentTile[northIdx], tempCurrentTile[southIdx], tempCurrentTile[southLowerIdx],
                tempCurrentTile[westLeftIdx], tempCurrentTile[westIdx], tempEastHaloZoneCurrent[i].first, tempEastHaloZoneCurrent[i].second,
                tempCurrentTile[centerIdx],
                domainParamsTile[northUpperIdx], domainParamsTile[northIdx], domainParamsTile[southIdx], domainParamsTile[southLowerIdx],
                domainParamsTile[westLeftIdx], domainParamsTile[westIdx], domainParamsEastHaloZone[i].first, domainParamsEastHaloZone[i].second,
                domainParamsTile[centerIdx],
                domainMapTile[centerIdx]);
        }
    }

    // left lower corner
    if (isNotBottomRow() && isNotLeftColumn()) // node is not in the bottom row and not in the left column
    {
        // the corner consists of 4 values, which must be stored to the tile, south and west halo zones
        int verticalIdx = 0;
        int haloZoneIdx = _sizes.localHeight - 1;
        
        // most left most lower value
        tempWestHaloZoneNext[haloZoneIdx].first = 
            tempHaloBotRow0Next[verticalIdx] = 
            tempTileBotRow0Next[verticalIdx] = computeCornerPoint.operator()<Corners::LEFT_LOWER, CornerPoints::LEFT_LOWER>();

        verticalIdx++;

        // second most left most lower value
        tempWestHaloZoneNext[haloZoneIdx].second = 
            tempHaloBotRow0Next[verticalIdx] = 
            tempTileBotRow0Next[verticalIdx] = computeCornerPoint.operator()<Corners::LEFT_LOWER, CornerPoints::RIGHT_LOWER>();

        verticalIdx--;
        haloZoneIdx--;

        // most left second most lower value
        tempWestHaloZoneNext[haloZoneIdx].first = 
            tempHaloBotRow1Next[verticalIdx] = 
            tempTileBotRow1Next[verticalIdx] = computeCornerPoint.operator()<Corners::LEFT_LOWER, CornerPoints::LEFT_UPPER>();

        verticalIdx++;

        // second most left second most lower value
        tempWestHaloZoneNext[haloZoneIdx].second = 
            tempHaloBotRow1Next[verticalIdx] = 
            tempTileBotRow1Next[verticalIdx] = computeCornerPoint.operator()<Corners::LEFT_LOWER, CornerPoints::RIGHT_UPPER>();
    }

    // south halo zone
    if (isNotBottomRow()) // node is not in the bottom row
    {
        for (size_t i = 2; i < _sizes.localWidth - 2; i++)
        {
            tempHaloBotRow0Next[i] = tempTileBotRow0Next[i] = computePoint(
                tempBotRow4Current[i], tempBotRow3Current[i], tempBotRow1Current[i], tempBotRow0Current[i],
                tempBotRow2Current[i - 2], tempBotRow2Current[i - 1], tempBotRow2Current[i + 1], tempBotRow2Current[i + 2],
                tempBotRow2Current[i],
                domainParamsBotRow4[i], domainParamsBotRow3[i], domainParamsBotRow1[i], domainParamsBotRow0[i],
                domainParamsBotRow2[i - 2], domainParamsBotRow2[i - 1], domainParamsBotRow2[i + 1], domainParamsBotRow2[i + 2],
                domainParamsBotRow2[i],
                domainMapBotCenter0[i]);
            tempHaloBotRow1Next[i] = tempTileBotRow1Next[i] = computePoint(
                tempBotRow5Current[i], tempBotRow4Current[i], tempBotRow2Current[i], tempBotRow1Current[i],
                tempBotRow3Current[i - 2], tempBotRow3Current[i - 1], tempBotRow3Current[i + 1], tempBotRow3Current[i + 2],
                tempBotRow3Current[i],
                domainParamsBotRow5[i], domainParamsBotRow4[i], domainParamsBotRow2[i], domainParamsBotRow1[i],
                domainParamsBotRow3[i - 2], domainParamsBotRow3[i - 1], domainParamsBotRow3[i + 1], domainParamsBotRow3[i + 2],
                domainParamsBotRow3[i],
                domainMapBotCenter1[i]);
        }
    }

    // right lower corner
    if (isNotBottomRow() && isNotRightColumn()) // node is not in the bottom row and not in the right column
    {
        // the corner consists of 4 values, which must be stored to the tile, south and east halo zones
        int verticalIdx = _sizes.localWidth - 2;
        int haloZoneIdx = _sizes.localHeight - 1;

        // second most right most lower value
        tempEastHaloZoneNext[haloZoneIdx].first = 
            tempHaloBotRow0Next[verticalIdx] = 
            tempTileBotRow0Next[verticalIdx] = computeCornerPoint.operator()<Corners::RIGHT_LOWER, CornerPoints::LEFT_LOWER>();

        verticalIdx++;

        // most right most lower value
        tempEastHaloZoneNext[haloZoneIdx].second = 
            tempHaloBotRow0Next[verticalIdx] = 
            tempTileBotRow0Next[verticalIdx] = computeCornerPoint.operator()<Corners::RIGHT_LOWER, CornerPoints::RIGHT_LOWER>();

        verticalIdx--;
        haloZoneIdx--;

        // second most right second most lower value
        tempEastHaloZoneNext[haloZoneIdx].first = 
            tempHaloBotRow1Next[verticalIdx] = 
            tempTileBotRow1Next[verticalIdx] = computeCornerPoint.operator()<Corners::RIGHT_LOWER, CornerPoints::LEFT_UPPER>();

        verticalIdx++;

        // most right second most lower value
        tempEastHaloZoneNext[haloZoneIdx].second = 
            tempHaloBotRow1Next[verticalIdx] = 
            tempTileBotRow1Next[verticalIdx] = computeCornerPoint.operator()<Corners::RIGHT_LOWER, CornerPoints::RIGHT_UPPER>();
    }
}

void ParallelHeatSolver::computeTempHaloZones_DataType(bool current, bool next)
{
    float *tempCurrentTile = _tempTiles[current].data();
    float *tempCurrentTopRow0 = _tempTiles[current].data();
    float *tempCurrentTopRow1 = _tempTiles[current].data() + _sizes.localWidthWithHalos;
    float *tempCurrentTopRow2 = _tempTiles[current].data() + 2 * _sizes.localWidthWithHalos;
    float *tempCurrentTopRow3 = _tempTiles[current].data() + 3 * _sizes.localWidthWithHalos;
    float *tempCurrentTopRow4 = _tempTiles[current].data() + 4 * _sizes.localWidthWithHalos;
    float *tempCurrentTopRow5 = _tempTiles[current].data() + 5 * _sizes.localWidthWithHalos;
    float *tempCurrentBotRow0 = _tempTiles[current].data() + (_sizes.localHeightWithHalos - 1) * _sizes.localWidthWithHalos; // closest to the bottom
    float *tempCurrentBotRow1 = _tempTiles[current].data() + (_sizes.localHeightWithHalos - 2) * _sizes.localWidthWithHalos;
    float *tempCurrentBotRow2 = _tempTiles[current].data() + (_sizes.localHeightWithHalos - 3) * _sizes.localWidthWithHalos;
    float *tempCurrentBotRow3 = _tempTiles[current].data() + (_sizes.localHeightWithHalos - 4) * _sizes.localWidthWithHalos;
    float *tempCurrentBotRow4 = _tempTiles[current].data() + (_sizes.localHeightWithHalos - 5) * _sizes.localWidthWithHalos;
    float *tempCurrentBotRow5 = _tempTiles[current].data() + (_sizes.localHeightWithHalos - 6) * _sizes.localWidthWithHalos;

    float *domainParamsTile = _domainParamsTile.data();
    float *domainParamsTopRow0 = _domainParamsTile.data();
    float *domainParamsTopRow1 = _domainParamsTile.data() + _sizes.localWidthWithHalos;
    float *domainParamsTopRow2 = _domainParamsTile.data() + 2 * _sizes.localWidthWithHalos;
    float *domainParamsTopRow3 = _domainParamsTile.data() + 3 * _sizes.localWidthWithHalos;
    float *domainParamsTopRow4 = _domainParamsTile.data() + 4 * _sizes.localWidthWithHalos;
    float *domainParamsTopRow5 = _domainParamsTile.data() + 5 * _sizes.localWidthWithHalos;
    float *domainParamsBotRow0 = _domainParamsTile.data() + (_sizes.localHeightWithHalos - 1) * _sizes.localWidthWithHalos; // closest to the bottom
    float *domainParamsBotRow1 = _domainParamsTile.data() + (_sizes.localHeightWithHalos - 2) * _sizes.localWidthWithHalos;
    float *domainParamsBotRow2 = _domainParamsTile.data() + (_sizes.localHeightWithHalos - 3) * _sizes.localWidthWithHalos;
    float *domainParamsBotRow3 = _domainParamsTile.data() + (_sizes.localHeightWithHalos - 4) * _sizes.localWidthWithHalos;
    float *domainParamsBotRow4 = _domainParamsTile.data() + (_sizes.localHeightWithHalos - 5) * _sizes.localWidthWithHalos;
    float *domainParamsBotRow5 = _domainParamsTile.data() + (_sizes.localHeightWithHalos - 6) * _sizes.localWidthWithHalos;

    int *domainMapTile = _domainMapTile.data();
    int *domainMapTopRow2 = _domainMapTile.data() + 2 * _sizes.localWidthWithHalos;
    int *domainMapTopRow3 = _domainMapTile.data() + 3 * _sizes.localWidthWithHalos;
    int *domainMapBotRow2 = _domainMapTile.data() + (_sizes.localHeightWithHalos - 3) * _sizes.localWidthWithHalos;
    int *domainMapBotRow3 = _domainMapTile.data() + (_sizes.localHeightWithHalos - 4) * _sizes.localWidthWithHalos;
    
    float *tempNextTile = _tempTiles[next].data();
    float *tempNextTopRow2 = _tempTiles[next].data() + 2 * _sizes.localWidthWithHalos;
    float *tempNextTopRow3 = _tempTiles[next].data() + 3 * _sizes.localWidthWithHalos;
    float *tempNextBotRow2 = _tempTiles[next].data() + (_sizes.localHeightWithHalos - 3) * _sizes.localWidthWithHalos;
    float *tempNextBotRow3 = _tempTiles[next].data() + (_sizes.localHeightWithHalos - 4) * _sizes.localWidthWithHalos;

    // north halo zone
    if (isNotTopRow())
    {
        for (size_t i = isNotLeftColumn() ? 2 : 4; i < _sizes.localWidthWithHalos - (isNotRightColumn() ? 2 : 4); i++)
        {
            tempNextTopRow2[i] = computePoint(
                tempCurrentTopRow0[i], tempCurrentTopRow1[i], tempCurrentTopRow3[i], tempCurrentTopRow4[i],
                tempCurrentTopRow2[i - 2], tempCurrentTopRow2[i - 1], tempCurrentTopRow2[i + 1], tempCurrentTopRow2[i + 2],
                tempCurrentTopRow2[i],
                domainParamsTopRow0[i], domainParamsTopRow1[i], domainParamsTopRow3[i], domainParamsTopRow4[i],
                domainParamsTopRow2[i - 2], domainParamsTopRow2[i - 1], domainParamsTopRow2[i + 1], domainParamsTopRow2[i + 2],
                domainParamsTopRow2[i],
                domainMapTopRow2[i]);
            tempNextTopRow3[i] = computePoint(
                tempCurrentTopRow1[i], tempCurrentTopRow2[i], tempCurrentTopRow4[i], tempCurrentTopRow5[i],
                tempCurrentTopRow3[i - 2], tempCurrentTopRow3[i - 1], tempCurrentTopRow3[i + 1], tempCurrentTopRow3[i + 2],
                tempCurrentTopRow3[i],
                domainParamsTopRow1[i], domainParamsTopRow2[i], domainParamsTopRow4[i], domainParamsTopRow5[i],
                domainParamsTopRow3[i - 2], domainParamsTopRow3[i - 1], domainParamsTopRow3[i + 1], domainParamsTopRow3[i + 2],
                domainParamsTopRow3[i],
                domainMapTopRow3[i]);
        }
    }

    // west zone
    if (isNotLeftColumn())
    {
        for (size_t i = 4; i < _sizes.localHeightWithHalos - 4; i++)
        {
            int northUpperIdx = (i - 2) * _sizes.localWidthWithHalos + 2;
            int northIdx = (i - 1) * _sizes.localWidthWithHalos + 2;
            int centerIdx = i * _sizes.localWidthWithHalos + 2;
            int southIdx = (i + 1) * _sizes.localWidthWithHalos + 2;
            int southLowerIdx = (i + 2) * _sizes.localWidthWithHalos + 2;
            int westLeftIdx = centerIdx - 2;
            int westIdx = centerIdx - 1;
            int eastIdx = centerIdx + 1;
            int eastRightIdx = centerIdx + 2;

            // west left
            tempNextTile[centerIdx] = computePoint(
                tempCurrentTile[northUpperIdx], tempCurrentTile[northIdx], tempCurrentTile[southIdx], tempCurrentTile[southLowerIdx],
                tempCurrentTile[westLeftIdx], tempCurrentTile[westIdx], tempCurrentTile[eastIdx], tempCurrentTile[eastRightIdx],
                tempCurrentTile[centerIdx],
                domainParamsTile[northUpperIdx], domainParamsTile[northIdx], domainParamsTile[southIdx], domainParamsTile[southLowerIdx],
                domainParamsTile[westLeftIdx], domainParamsTile[westIdx], domainParamsTile[eastIdx], domainParamsTile[eastRightIdx],
                domainParamsTile[centerIdx],
                domainMapTile[centerIdx]);
            
            northUpperIdx++;
            northIdx++;
            centerIdx++;
            southIdx++;
            southLowerIdx++;
            westLeftIdx++;
            westIdx++;
            eastIdx++;
            eastRightIdx++;

            // west right
            tempNextTile[centerIdx] = computePoint(
                tempCurrentTile[northUpperIdx], tempCurrentTile[northIdx], tempCurrentTile[southIdx], tempCurrentTile[southLowerIdx],
                tempCurrentTile[westLeftIdx], tempCurrentTile[westIdx], tempCurrentTile[eastIdx], tempCurrentTile[eastRightIdx],
                tempCurrentTile[centerIdx],
                domainParamsTile[northUpperIdx], domainParamsTile[northIdx], domainParamsTile[southIdx], domainParamsTile[southLowerIdx],
                domainParamsTile[westLeftIdx], domainParamsTile[westIdx], domainParamsTile[eastIdx], domainParamsTile[eastRightIdx],
                domainParamsTile[centerIdx],
                domainMapTile[centerIdx]
            );
        }
    }

    // east zone
    if (isNotRightColumn())
    {
        for (size_t i = 4; i < _sizes.localHeightWithHalos - 4; i++)
        {
            int northUpperIdx = (i - 1) * _sizes.localWidthWithHalos - 4;
            int northIdx = i * _sizes.localWidthWithHalos - 4;
            int centerIdx = (i + 1) * _sizes.localWidthWithHalos - 4;
            int southIdx = (i + 2) * _sizes.localWidthWithHalos - 4;
            int southLowerIdx = (i + 3) * _sizes.localWidthWithHalos - 4;
            int westLeftIdx = centerIdx - 2;
            int westIdx = centerIdx - 1;
            int eastIdx = centerIdx + 1;
            int eastRightIdx = centerIdx + 2;

            // east left
            tempNextTile[centerIdx] = computePoint(
                tempCurrentTile[northUpperIdx], tempCurrentTile[northIdx], tempCurrentTile[southIdx], tempCurrentTile[southLowerIdx],
                tempCurrentTile[westLeftIdx], tempCurrentTile[westIdx], tempCurrentTile[eastIdx], tempCurrentTile[eastRightIdx],
                tempCurrentTile[centerIdx],
                domainParamsTile[northUpperIdx], domainParamsTile[northIdx], domainParamsTile[southIdx], domainParamsTile[southLowerIdx],
                domainParamsTile[westLeftIdx], domainParamsTile[westIdx], domainParamsTile[eastIdx], domainParamsTile[eastRightIdx],
                domainParamsTile[centerIdx],
                domainMapTile[centerIdx]);
            
            northUpperIdx++;
            northIdx++;
            centerIdx++;
            southIdx++;
            southLowerIdx++;
            westLeftIdx++;
            westIdx++;
            eastIdx++;
            eastRightIdx++;
            
            // east right
            tempNextTile[centerIdx] = computePoint(
                tempCurrentTile[northUpperIdx], tempCurrentTile[northIdx], tempCurrentTile[southIdx], tempCurrentTile[southLowerIdx],
                tempCurrentTile[westLeftIdx], tempCurrentTile[westIdx], tempCurrentTile[eastIdx], tempCurrentTile[eastRightIdx],
                tempCurrentTile[centerIdx],
                domainParamsTile[northUpperIdx], domainParamsTile[northIdx], domainParamsTile[southIdx], domainParamsTile[southLowerIdx],
                domainParamsTile[westLeftIdx], domainParamsTile[westIdx], domainParamsTile[eastIdx], domainParamsTile[eastRightIdx],
                domainParamsTile[centerIdx],
                domainMapTile[centerIdx]
            );
        }
    }

    // south halo zone
    if (isNotBottomRow())
    {
        for (size_t i = isNotLeftColumn() ? 2 : 4; i < _sizes.localWidthWithHalos - (isNotRightColumn() ? 2 : 4); i++)
        {
            tempNextBotRow2[i] = computePoint(
                tempCurrentBotRow4[i], tempCurrentBotRow3[i], tempCurrentBotRow1[i], tempCurrentBotRow0[i],
                tempCurrentBotRow2[i - 2], tempCurrentBotRow2[i - 1], tempCurrentBotRow2[i + 1], tempCurrentBotRow2[i + 2],
                tempCurrentBotRow2[i],
                domainParamsBotRow4[i], domainParamsBotRow3[i], domainParamsBotRow1[i], domainParamsBotRow0[i],
                domainParamsBotRow2[i - 2], domainParamsBotRow2[i - 1], domainParamsBotRow2[i + 1], domainParamsBotRow2[i + 2],
                domainParamsBotRow2[i],
                domainMapBotRow2[i]);
            tempNextBotRow3[i] = computePoint(
                tempCurrentBotRow5[i], tempCurrentBotRow4[i], tempCurrentBotRow2[i], tempCurrentBotRow1[i],
                tempCurrentBotRow3[i - 2], tempCurrentBotRow3[i - 1], tempCurrentBotRow3[i + 1], tempCurrentBotRow3[i + 2],
                tempCurrentBotRow3[i],
                domainParamsBotRow5[i], domainParamsBotRow4[i], domainParamsBotRow2[i], domainParamsBotRow1[i],
                domainParamsBotRow3[i - 2], domainParamsBotRow3[i - 1], domainParamsBotRow3[i + 1], domainParamsBotRow3[i + 2],
                domainParamsBotRow3[i],
                domainMapBotRow3[i]);
        }
    }
}

void ParallelHeatSolver::computeTempTile_Raw(bool current, bool next)
{
    float *tempCurrentTile = _tempTiles[current].data();
    float *tempNextTile = _tempTiles[next].data();

    for (size_t i = 2; i < _sizes.localHeight - 2; i++)
    {
        const int northUpperIdx = (i - 2) * _sizes.localWidth;
        const int northIdx = (i - 1) * _sizes.localWidth;
        const int centerIdx = i * _sizes.localWidth;
        const int southIdx = (i + 1) * _sizes.localWidth;
        const int southLowerIdx = (i + 2) * _sizes.localWidth;
        const int westLeftIdx = i * _sizes.localWidth - 2;
        const int westIdx = i * _sizes.localWidth - 1;
        const int eastIdx = i * _sizes.localWidth + 1;
        const int eastRightIdx = i * _sizes.localWidth + 2;

        for (size_t j = 2; j < _sizes.localWidth - 2; j++)
        {
            tempNextTile[centerIdx + j] = computePoint(
                tempCurrentTile[northUpperIdx + j], tempCurrentTile[northIdx + j], tempCurrentTile[southIdx + j], tempCurrentTile[southLowerIdx + j],
                tempCurrentTile[westLeftIdx + j], tempCurrentTile[westIdx + j], tempCurrentTile[eastIdx + j], tempCurrentTile[eastRightIdx + j],
                tempCurrentTile[centerIdx + j],
                _domainParamsTile[northUpperIdx + j], _domainParamsTile[northIdx + j], _domainParamsTile[southIdx + j], _domainParamsTile[southLowerIdx + j],
                _domainParamsTile[westLeftIdx + j], _domainParamsTile[westIdx + j], _domainParamsTile[eastIdx + j], _domainParamsTile[eastRightIdx + j],
                _domainParamsTile[centerIdx + j],
                _domainMapTile[centerIdx + j]);
        }
    }
}

void ParallelHeatSolver::computeTempTile_DataType(bool current, bool next)
{
    float *tempCurrentTile = _tempTiles[current].data();
    float *tempNextTile = _tempTiles[next].data();
    float *domainParamsTile = _domainParamsTile.data();
    int *domainMapTile = _domainMapTile.data();

    for (size_t i = 4; i < _sizes.localHeightWithHalos - 4; i++)
    {
        const int northUpperIdx = (i - 2) * _sizes.localWidthWithHalos;
        const int northIdx = (i - 1) * _sizes.localWidthWithHalos;
        const int centerIdx = i * _sizes.localWidthWithHalos;
        const int southIdx = (i + 1) * _sizes.localWidthWithHalos;
        const int southLowerIdx = (i + 2) * _sizes.localWidthWithHalos;
        const int westLeftIdx = centerIdx - 2;
        const int westIdx = centerIdx - 1;
        const int eastIdx = centerIdx + 1;
        const int eastRightIdx = centerIdx + 2;

        for (size_t j = 4; j < _sizes.localWidthWithHalos - 4; j++)
        {
            tempNextTile[centerIdx + j] = computePoint(
                tempCurrentTile[northUpperIdx + j], tempCurrentTile[northIdx + j], tempCurrentTile[southIdx + j], tempCurrentTile[southLowerIdx + j],
                tempCurrentTile[westLeftIdx + j], tempCurrentTile[westIdx + j], tempCurrentTile[eastIdx + j], tempCurrentTile[eastRightIdx + j],
                tempCurrentTile[centerIdx + j],
                domainParamsTile[northUpperIdx + j], domainParamsTile[northIdx + j], domainParamsTile[southIdx + j], domainParamsTile[southLowerIdx + j],
                domainParamsTile[westLeftIdx + j], domainParamsTile[westIdx + j], domainParamsTile[eastIdx + j], domainParamsTile[eastRightIdx + j],
                domainParamsTile[centerIdx + j],
                domainMapTile[centerIdx + j]);
        }
    }
}

void ParallelHeatSolver::computeAndPrintMidColAverageParallel_Raw(size_t iteration)
{
    float sum = 0;
    pair<float, float> *tempWestHaloZoneNext = reinterpret_cast<pair<float, float> *>(_tempHaloZones[1].data() + 2 * _sizes.northSouthHalo);
    if (_decomposition.nx == 1) // there is only one column, middle column of the tile must be sampled (can happen only with 1 or 2 processes)
    {
        int midColIdx = _sizes.localWidth >> 1;
        for (size_t i = 0; i < _sizes.localHeight; i++)
        {
            sum += _tempTiles[1][i * _sizes.localWidth + midColIdx];
        }
        sum /= _sizes.localHeight;
    }
    else // there are multiple columns, left most column must be sampled
    {
        for (size_t i = 0; i < _sizes.localHeight; i++)
        {
            sum += tempWestHaloZoneNext[i].first;
        }
        sum /= _sizes.localHeight;
    }

    float reducedSum = 0;
    MPI_Reduce(&sum, &reducedSum, 1, MPI_FLOAT, MPI_SUM, 0, _midColComm);

    if (_midColRank == 0)
    {
        reducedSum /= _decomposition.ny;
        printProgressReport(iteration, reducedSum);
    }
}

void ParallelHeatSolver::computeAndPrintMidColAverageParallel_DataType(size_t iteration)
{
    float sum = 0;
    pair<float, float> *tempWestHaloZoneNext = reinterpret_cast<pair<float, float> *>(_tempHaloZones[1].data() + 2 * _sizes.northSouthHalo);
    if (_decomposition.nx == 1) // there is only one column, middle column of the tile must be sampled (can happen only with 1 or 2 processes)
    {
        int midColIdx = _sizes.localWidthWithHalos >> 1;
        for (size_t i = 2; i < _sizes.localHeightWithHalos - 2; i++)
        {
            sum += _tempTiles[!(iteration & 1)][i * _sizes.localWidthWithHalos + midColIdx + 2];
        }
        sum /= _sizes.localHeight;
    }
    else // there are multiple columns, left most column must be sampled
    {
        for (size_t i = 2; i < _sizes.localHeightWithHalos - 2; i++)
        {
            sum += _tempTiles[!(iteration & 1)][i * _sizes.localWidthWithHalos + 1];
        }
        sum /= _sizes.localHeight;
    }

    float reducedSum = 0;
    MPI_Reduce(&sum, &reducedSum, 1, MPI_FLOAT, MPI_SUM, 0, _midColComm);

    if (_midColRank == 0)
    {
        reducedSum /= _decomposition.ny;
        printProgressReport(iteration, reducedSum);
    }
}

void ParallelHeatSolver::computeAndPrintMidColAverageSequential(float timeElapsed, const vector<float, AlignedAllocator<float>> &outResult)
{
    if (_worldRank == 0)
    {
        float averageTemp = 0;
        for (size_t i = 0; i < mMaterialProps.getEdgeSize(); i++)
        {
            averageTemp += outResult[i * mMaterialProps.getEdgeSize() + (mMaterialProps.getEdgeSize() >> 1)];
        }
        averageTemp /= mMaterialProps.getEdgeSize();

        printFinalReport(timeElapsed, averageTemp);
    }
}

void ParallelHeatSolver::startHaloExchangeP2P_Raw()
{
    // leverage the created 2D topology to exchange the halo zones
    MPI_Ineighbor_alltoallv(_tempHaloZones[1].data(), _transferCounts, _displacements, MPI_FLOAT,
                            _tempHaloZones[0].data(), _transferCounts, _displacements, MPI_FLOAT, _topologyComm, 
                            &_haloExchangeRequest);
}

void ParallelHeatSolver::startHaloExchangeP2P_DataType(bool next)
{
    MPI_Ineighbor_alltoallw(_tempTiles[next].data(), _transferCountsDataType, _displacementsDataType, _floatSendHaloZoneTypes,
                            _tempTiles[next].data(), _transferCountsDataType, _displacementsDataType, _floatRecvHaloZoneTypes, _topologyComm, 
                            &_haloExchangeRequest);
}

void ParallelHeatSolver::startHaloExchangeRMA_Raw()
{
    MPI_Win_fence(0, _haloExchangeWindows[0]); // synchronize the access to the window

    for (int i = 0; i < 4; i++) // north, south, west, east
    {
        if (_neighbors[i] != MPI_PROC_NULL) // if the neighbor exists (not a boundary node)
        {
            MPI_Put(_tempHaloZones[1].data() + _displacements[i], _transferCounts[i], MPI_FLOAT,
                    _neighbors[i], _inverseDisplacements[i], _transferCounts[i], MPI_FLOAT, _haloExchangeWindows[0]);
        }
    }
}

void ParallelHeatSolver::startHaloExchangeRMA_DataType(bool next)
{
    MPI_Win_fence(0, _haloExchangeWindows[next]); // synchronize the access to the window

    for (int i = 0; i < 4; i++) // north, south, west, east
    {
        if (_neighbors[i] != MPI_PROC_NULL) // if the neighbor exists (not a boundary node)
        {
            MPI_Put(_tempTiles[next].data(), 1, _floatSendHaloZoneTypes[i],
                    _neighbors[i], 0, 1, _floatInverseRecvHaloZoneTypes[i], _haloExchangeWindows[next]);
        }
    }
}

void ParallelHeatSolver::awaitHaloExchangeP2P_Raw()
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

    // int actualRecvCount;
    // MPI_Get_count(&status, MPI_FLOAT, &actualRecvCount);
    // if (actualRecvCount != _sizes.northSouthHalo * 2 + _sizes.westEastHalo * 2)
    //{
    //     cerr << "Rank: " << _worldRank << " - Error in halo exchange: received unexpected number of values. Expected: " <<  _sizes.northSouthHalo * 2 + _sizes.westEastHalo * 2  << " Received: " << actualRecvCount << endl;
    //     MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COUNT);
    // }
}

void ParallelHeatSolver::awaitHaloExchangeP2P_DataType(bool next)
{
    (void)next;
    awaitHaloExchangeP2P_Raw();
}

void ParallelHeatSolver::awaitHaloExchangeRMA_Raw()
{
    MPI_Win_fence(0, _haloExchangeWindows[0]);
}

void ParallelHeatSolver::awaitHaloExchangeRMA_DataType(bool next)
{
    MPI_Win_fence(0, _haloExchangeWindows[next]);
}

void ParallelHeatSolver::scatterInitialData_Raw()
{
    MPI_Scatterv(mMaterialProps.getInitialTemperature().data(), _scatterGatherCounts.data(), _scatterGatherDisplacements.data(), _floatTileWithoutHaloZonesResized,
                 _tempTiles[0].data(), _sizes.localTile, MPI_FLOAT, 0, MPI_COMM_WORLD);
     
    MPI_Scatterv(mMaterialProps.getDomainParameters().data(), _scatterGatherCounts.data(), _scatterGatherDisplacements.data(), _floatTileWithoutHaloZonesResized,
                 _domainParamsTile.data(), _sizes.localTile, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
    MPI_Scatterv(mMaterialProps.getDomainMap().data(), _scatterGatherCounts.data(), _scatterGatherDisplacements.data(), _intTileWithoutHaloZonesResized,
                 _domainMapTile.data(), _sizes.localTile, MPI_INT, 0, MPI_COMM_WORLD);

    // copy to halo zones North and South
    copy(_tempTiles[0].begin(), _tempTiles[0].begin() + _sizes.northSouthHalo, _tempHaloZones[1].begin());                     // North
    copy(_tempTiles[0].end() - _sizes.northSouthHalo, _tempTiles[0].end(), _tempHaloZones[1].begin() + _sizes.northSouthHalo); // South

    copy(_domainParamsTile.begin(), _domainParamsTile.begin() + _sizes.northSouthHalo, _domainParamsHaloZoneTmp.begin());                     // North
    copy(_domainParamsTile.end() - _sizes.northSouthHalo, _domainParamsTile.end(), _domainParamsHaloZoneTmp.begin() + _sizes.northSouthHalo); // South

    // copy to halo zones West and East
    for (size_t i = 0; i < _sizes.localHeight; i++)
    {
        _tempHaloZones[1][2 * _sizes.northSouthHalo + 2 * i] = _tempTiles[0][i * _sizes.localWidth];                                                   // West
        _tempHaloZones[1][2 * _sizes.northSouthHalo + 2 * i + 1] = _tempTiles[0][i * _sizes.localWidth + 1];                                           // West
        _tempHaloZones[1][2 * _sizes.northSouthHalo + _sizes.westEastHalo + 2 * i] = _tempTiles[0][i * _sizes.localWidth + _sizes.localWidth - 2];     // East
        _tempHaloZones[1][2 * _sizes.northSouthHalo + _sizes.westEastHalo + 2 * i + 1] = _tempTiles[0][i * _sizes.localWidth + _sizes.localWidth - 1]; // East

        _domainParamsHaloZoneTmp[2 * _sizes.northSouthHalo + 2 * i] = _domainParamsTile[i * _sizes.localWidth];                                                         // West
        _domainParamsHaloZoneTmp[2 * _sizes.northSouthHalo + 2 * i + 1] = _domainParamsTile[i * _sizes.localWidth + 1];                                                 // West
        _domainParamsHaloZoneTmp[2 * _sizes.northSouthHalo + _sizes.westEastHalo + 2 * i] = _domainParamsTile[i * _sizes.localWidth + _sizes.localWidth - 2];     // East
        _domainParamsHaloZoneTmp[2 * _sizes.northSouthHalo + _sizes.westEastHalo + 2 * i + 1] = _domainParamsTile[i * _sizes.localWidth + _sizes.localWidth - 1]; // East
    }

    // exchange the halo zones
    MPI_Neighbor_alltoallv(_tempHaloZones[1].data(), _transferCounts, _displacements, MPI_FLOAT,
                           _tempHaloZones[0].data(), _transferCounts, _displacements, MPI_FLOAT, _topologyComm);
    MPI_Neighbor_alltoallv(_domainParamsHaloZoneTmp.data(), _transferCounts, _displacements, MPI_FLOAT,
                           _domainParamsHaloZone.data(), _transferCounts, _displacements, MPI_FLOAT, _topologyComm);

    // copy initial temperature to the second buffer
    copy(_tempTiles[0].begin(), _tempTiles[0].end(), _tempTiles[1].begin());
}

void ParallelHeatSolver::scatterInitialData_DataType()
{
    MPI_Scatterv(mMaterialProps.getInitialTemperature().data(), _scatterGatherCounts.data(), _scatterGatherDisplacements.data(), _floatTileWithoutHaloZonesResized,
                 _tempTiles[0].data(), 1, _floatTileWithHaloZones, 0, MPI_COMM_WORLD);
    MPI_Neighbor_alltoallw(_tempTiles[0].data(), _transferCountsDataType, _displacementsDataType, _floatSendHaloZoneTypes,
                           _tempTiles[0].data(), _transferCountsDataType, _displacementsDataType, _floatRecvHaloZoneTypes, _topologyComm);
    copy(_tempTiles[0].begin(), _tempTiles[0].end(), _tempTiles[1].begin());
    
    MPI_Scatterv(mMaterialProps.getDomainParameters().data(), _scatterGatherCounts.data(), _scatterGatherDisplacements.data(), _floatTileWithoutHaloZonesResized,
                 _domainParamsTile.data(), 1, _floatTileWithHaloZones, 0, MPI_COMM_WORLD);
    MPI_Neighbor_alltoallw(_domainParamsTile.data(), _transferCountsDataType, _displacementsDataType, _floatSendHaloZoneTypes,
                           _domainParamsTile.data(), _transferCountsDataType, _displacementsDataType, _floatRecvHaloZoneTypes, _topologyComm);
    
    MPI_Scatterv(mMaterialProps.getDomainMap().data(), _scatterGatherCounts.data(), _scatterGatherDisplacements.data(), _intTileWithoutHaloZonesResized,
                 _domainMapTile.data(), 1, _intTileWithHaloZones, 0, MPI_COMM_WORLD);
}

void ParallelHeatSolver::gatherComputedTempData_Raw(bool tile, vector<float, AlignedAllocator<float>> &outResult)
{
    MPI_Gatherv(_tempTiles[tile].data(), _sizes.localTile, MPI_FLOAT,
                outResult.data(), _scatterGatherCounts.data(), _scatterGatherDisplacements.data(), _floatTileWithoutHaloZonesResized, 0, MPI_COMM_WORLD);
}

void ParallelHeatSolver::gatherComputedTempData_DataType(bool tile, vector<float, AlignedAllocator<float>> &outResult)
{
    MPI_Gatherv(_tempTiles[tile].data(), 1, _floatTileWithHaloZones,
                outResult.data(), _scatterGatherCounts.data(), _scatterGatherDisplacements.data(), _floatTileWithoutHaloZonesResized, 0, MPI_COMM_WORLD);
}

void ParallelHeatSolver::prepareInitialHaloZones()
{
    // copy to halo zones North and South
    // use the odd halo zone, so that after the initial exchange, the even halo zone can be used for the first iteration
    copy(_tempTiles[0].begin(), _tempTiles[0].begin() + _sizes.northSouthHalo, _tempHaloZones[1].begin());                       // North
    copy(_tempTiles[0].end() - _sizes.northSouthHalo, _tempTiles[0].end(), _tempHaloZones[1].begin() + _sizes.northSouthHalo); // South

    copy(_domainParamsTile.begin(), _domainParamsTile.begin() + _sizes.northSouthHalo, _domainParamsHaloZoneTmp.begin());                       // North
    copy(_domainParamsTile.end() - _sizes.northSouthHalo, _domainParamsTile.end(), _domainParamsHaloZoneTmp.begin() + _sizes.northSouthHalo); // South

    // copy to halo zones West and East
    for (size_t i = 0; i < _sizes.localHeight; i++)
    {
        // use the odd halo zone, so that after the initial exchange, the even halo zone can be used for the first iteration
        _tempHaloZones[1][2 * _sizes.northSouthHalo + 2 * i] = _tempTiles[0][i * _sizes.localWidth];                                                         // West
        _tempHaloZones[1][2 * _sizes.northSouthHalo + 2 * i + 1] = _tempTiles[0][i * _sizes.localWidth + 1];                                                 // West
        _tempHaloZones[1][2 * _sizes.northSouthHalo + _sizes.westEastHalo + 2 * i] = _tempTiles[0][i * _sizes.localWidth + _sizes.localWidth - 2];     // East
        _tempHaloZones[1][2 * _sizes.northSouthHalo + _sizes.westEastHalo + 2 * i + 1] = _tempTiles[0][i * _sizes.localWidth + _sizes.localWidth - 1]; // East

        _domainParamsHaloZoneTmp[2 * _sizes.northSouthHalo + 2 * i] = _domainParamsTile[i * _sizes.localWidth];                                                         // West
        _domainParamsHaloZoneTmp[2 * _sizes.northSouthHalo + 2 * i + 1] = _domainParamsTile[i * _sizes.localWidth + 1];                                                 // West
        _domainParamsHaloZoneTmp[2 * _sizes.northSouthHalo + _sizes.westEastHalo + 2 * i] = _domainParamsTile[i * _sizes.localWidth + _sizes.localWidth - 2];     // East
        _domainParamsHaloZoneTmp[2 * _sizes.northSouthHalo + _sizes.westEastHalo + 2 * i + 1] = _domainParamsTile[i * _sizes.localWidth + _sizes.localWidth - 1]; // East
    }
}

void ParallelHeatSolver::exchangeInitialHaloZones()
{
    // scatter/gather the halo zones across neighbors
    MPI_Neighbor_alltoallv(_tempHaloZones[1].data(), _transferCounts, _displacements, MPI_FLOAT,
                           _tempHaloZones[0].data(), _transferCounts, _displacements, MPI_FLOAT, _topologyComm);
    MPI_Neighbor_alltoallv(_domainParamsHaloZoneTmp.data(), _transferCounts, _displacements, MPI_FLOAT,
                           _domainParamsHaloZone.data(), _transferCounts, _displacements, MPI_FLOAT, _topologyComm);
}

void ParallelHeatSolver::run(vector<float, AlignedAllocator<float>> &outResult)
{
    if (_worldRank == 0)
    {
        outResult.resize(_sizes.globalEdge * _sizes.globalEdge);
    }

#if DATA_TYPE_EXCHANGE
    scatterInitialData_DataType();
    /*for (int i = 0; i < _worldSize; i++)
    {
        printTile(i, _tempTiles[0], 4);
        printTile(i, _tempTiles[1], 4);
    }*/
    /*for (int i = 0; i < _worldSize; i++)
    {
        printTile(i, _tempTiles[1], 4);
    }*/
    /*for (int i = 0; i < _worldSize; i++)
    {
        printTile(i, _domainParamsTile, 4);
    }*/
    //return;

    const auto startHaloExchangeFunction = mSimulationProps.isRunParallelRMA() ? &ParallelHeatSolver::startHaloExchangeRMA_DataType : &ParallelHeatSolver::startHaloExchangeP2P_DataType;
    const auto awaitHaloExchangeFunction = mSimulationProps.isRunParallelRMA() ? &ParallelHeatSolver::awaitHaloExchangeRMA_DataType : &ParallelHeatSolver::awaitHaloExchangeP2P_DataType;

    double startTime = MPI_Wtime();
    // run the simulation
    for (size_t iter = 0; iter < mSimulationProps.getNumIterations(); iter++)
    {
        const bool current = iter & 1;
        const bool next = !current;

        // compute temperature halo zones (and the two most outer rows and columns of the tile)
        computeTempHaloZones_DataType(current, next);

        // start the halo zone exchange (async P2P communication, or RMA communication if enabled)
        (this->*startHaloExchangeFunction)(next);

        // compute the rest of the tile (inner part)
        computeTempTile_DataType(current, next);

        // wait for all halo zone exchanges to finalize
        (this->*awaitHaloExchangeFunction)(next);

        /*if (iter == 3)
        {
            for (int i = 0; i < _worldSize; i++)
            {
                printTile(i, _tempTiles[next], 4);
            }
            return;
        }*/

        if (shouldStoreData(iter))
        {
            if (mSimulationProps.useParallelIO())
            {
                storeDataIntoFileParallel(iter, 2, _tempTiles[next].data());
            }
            else
            {
                gatherComputedTempData_DataType(next, outResult);
                if (_worldRank == 0)
                {
                    storeDataIntoFileSequential(iter, outResult.data());
                }
            }
        }

        if (shouldPrintProgress(iter) && shouldComputeMiddleColumnAverageTemperature())
        {
            // compute and print the middle column average temperature using reduction of partial averages
            computeAndPrintMidColAverageParallel_DataType(iter);
        }
    }
    double elapsedTime = MPI_Wtime() - startTime;

    // retrieve the final temperature from all the nodes to a single matrix
    gatherComputedTempData_DataType(mSimulationProps.getNumIterations() & 1, outResult);

    // compute the final average temperature and print the final report
    computeAndPrintMidColAverageSequential(elapsedTime, outResult);

#elif RAW_EXCHANGE
    // scatter the initial data across the nodes from the root node
    scatterInitialData_Raw();

    // deallocate no longer needed temporary buffers
    _domainParamsHaloZoneTmp.resize(0);
    _initialScatterDomainParams.resize(0);
    _initialScatterDomainMap.resize(0);

    const auto startHaloExchangeFunction = mSimulationProps.isRunParallelRMA() ? &ParallelHeatSolver::startHaloExchangeRMA_Raw : &ParallelHeatSolver::startHaloExchangeP2P_Raw;
    const auto awaitHaloExchangeFunction = mSimulationProps.isRunParallelRMA() ? &ParallelHeatSolver::awaitHaloExchangeRMA_Raw : &ParallelHeatSolver::awaitHaloExchangeP2P_Raw;

    double startTime = MPI_Wtime();

    // run the simulation
    for (size_t iter = 0; iter < mSimulationProps.getNumIterations(); iter++)
    {
        const bool current = iter & 1;
        const bool next = !current;

        // compute temperature halo zones (and the two most outer rows and columns of the tile)
        computeTempHaloZones_Raw(current, next);

        // start the halo zone exchange (async P2P communication, or RMA communication if enabled)
        (this->*startHaloExchangeFunction)();

        // compute the rest of the tile (inner part)
        computeTempTile_Raw(current, next);

        // wait for all halo zone exchanges to finalize
        (this->*awaitHaloExchangeFunction)();

        if (shouldStoreData(iter))
        {
            if (mSimulationProps.useParallelIO())
            {
                storeDataIntoFileParallel(iter, 0, _tempTiles[next].data());
            }
            else
            {
                gatherComputedTempData_Raw(next, outResult);
                if (_worldRank == 0)
                {
                    storeDataIntoFileSequential(iter, outResult.data());
                }
            }
        }

        if (shouldPrintProgress(iter) && shouldComputeMiddleColumnAverageTemperature())
        {
            // compute and print the middle column average temperature using reduction of partial averages
            computeAndPrintMidColAverageParallel_Raw(iter);
        }
    }
    double elapsedTime = MPI_Wtime() - startTime;

    // retrieve the final temperature from all the nodes to a single matrix
    gatherComputedTempData_Raw(mSimulationProps.getNumIterations() & 1, outResult);

    // compute the final average temperature and print the final report
    computeAndPrintMidColAverageSequential(elapsedTime, outResult);
#endif
}

bool ParallelHeatSolver::shouldComputeMiddleColumnAverageTemperature() const
{
    return _midColComm != MPI_COMM_NULL;
}

void ParallelHeatSolver::openOutputFileSequential()
{
    // Create the output file for sequential access.
    mFileHandle = H5Fcreate(mSimulationProps.getOutputFileName(codeType).c_str(),
                            H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (!mFileHandle.valid())
    {
        throw ios::failure("Cannot create output file!");
    }
}

void ParallelHeatSolver::storeDataIntoFileSequential(size_t iteration, const float *globalData)
{
    storeDataIntoFile(mFileHandle, iteration, globalData);
}

void ParallelHeatSolver::openOutputFileParallel()
{
#ifdef H5_HAVE_PARALLEL
    Hdf5PropertyListHandle faplHandle(H5Pcreate(H5P_FILE_ACCESS));
    H5Pset_fapl_mpio(faplHandle, _topologyComm, MPI_INFO_NULL);

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
        throw ios::failure("Cannot create output file!");
    }
#else
    throw runtime_error("Parallel HDF5 support is not available!");
#endif /* H5_HAVE_PARALLEL */
}

void ParallelHeatSolver::storeDataIntoFileParallel(size_t iteration, int halloOffset, const float *localData)
{
    if (mFileHandle == H5I_INVALID_HID)
    {
        return;
    }

#ifdef H5_HAVE_PARALLEL
    array<hsize_t, 2> gridSizes{static_cast<hsize_t>(mMaterialProps.getEdgeSize()), static_cast<hsize_t>(mMaterialProps.getEdgeSize())};
    array<hsize_t, 2> localGridSizes{static_cast<hsize_t>(_sizes.localHeight + 2 * halloOffset), static_cast<hsize_t>(_sizes.localWidth + 2 * halloOffset)};
    array<hsize_t, 2> localGridSizesWithoutHalo{static_cast<hsize_t>(_sizes.localHeight), static_cast<hsize_t>(_sizes.localWidth)};
    array<hsize_t, 2> tileOffsets{static_cast<hsize_t>(_sizes.localHeight * (_worldRank / _decomposition.nx)),
                                  static_cast<hsize_t>(_sizes.localWidth * (_worldRank % _decomposition.nx))};
    array<hsize_t, 2> halloOffsets{halloOffset, halloOffset};
    array<hsize_t, 2> blockCounts{1, 1};

    // Create new HDF5 group in the output file
    string groupName = "Timestep_" + to_string(iteration / mSimulationProps.getWriteIntensity());

    Hdf5GroupHandle groupHandle(H5Gcreate(mFileHandle, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    
    {
        /**********************************************************************************************************************/
        /*                                Compute the tile offsets and sizes.                                                 */
        /*               Note that the X and Y coordinates are swapped (but data not altered).                                */
        /**********************************************************************************************************************/

        // Create new dataspace and dataset using it.
        static constexpr string_view dataSetName{"Temperature"};

        Hdf5PropertyListHandle datasetPropListHandle(H5Pcreate(H5P_DATASET_CREATE));
        H5Pset_chunk(datasetPropListHandle, 2, localGridSizesWithoutHalo.data());
        /**********************************************************************************************************************/
        /*                            Create dataset property list to set up chunking.                                        */
        /*                Set up chunking for collective write operation in datasetPropListHandle variable.                   */
        /**********************************************************************************************************************/

        Hdf5DataspaceHandle dataSpaceHandle(H5Screate_simple(2, gridSizes.data(), nullptr));
        Hdf5DatasetHandle dataSetHandle(H5Dcreate(groupHandle, dataSetName.data(),
                                                  H5T_NATIVE_FLOAT, dataSpaceHandle,
                                                  H5P_DEFAULT, datasetPropListHandle,
                                                  H5P_DEFAULT));
        H5Sselect_hyperslab(dataSpaceHandle, H5S_SELECT_SET, tileOffsets.data(), nullptr, blockCounts.data(), localGridSizesWithoutHalo.data());

        Hdf5DataspaceHandle memSpaceHandle(H5Screate_simple(2, localGridSizes.data(), nullptr));
        H5Sselect_hyperslab(memSpaceHandle, H5S_SELECT_SET, halloOffsets.data(), nullptr, blockCounts.data(), localGridSizesWithoutHalo.data());

        /**********************************************************************************************************************/
        /*                Create memory dataspace representing tile in the memory (set up memSpaceHandle).                    */
        /**********************************************************************************************************************/

        /**********************************************************************************************************************/
        /*              Select inner part of the tile in memory and matching part of the dataset in the file                  */
        /*                           (given by position of the tile in global domain).                                        */
        /**********************************************************************************************************************/


        Hdf5PropertyListHandle propListHandle(H5Pcreate(H5P_DATASET_XFER));
        H5Pset_dxpl_mpio(propListHandle, H5FD_MPIO_COLLECTIVE);

        /**********************************************************************************************************************/
        /*              Perform collective write operation, writting tiles from all processes at once.                        */
        /*                                   Set up the propListHandle variable.                                              */
        /**********************************************************************************************************************/

        H5Dwrite(dataSetHandle, H5T_NATIVE_FLOAT, memSpaceHandle, dataSpaceHandle, propListHandle, localData);
    }

    {
        // 3. Store attribute with current iteration number in the group.
        static constexpr string_view attributeName{"Time"};
        Hdf5DataspaceHandle dataSpaceHandle(H5Screate(H5S_SCALAR));
        Hdf5AttributeHandle attributeHandle(H5Acreate2(groupHandle, attributeName.data(),
                                                       H5T_IEEE_F64LE, dataSpaceHandle,
                                                       H5P_DEFAULT, H5P_DEFAULT));
        const double snapshotTime = static_cast<double>(iteration);
        H5Awrite(attributeHandle, H5T_IEEE_F64LE, &snapshotTime);
    }
#else
    throw runtime_error("Parallel HDF5 support is not available!");
#endif /* H5_HAVE_PARALLEL */
}
