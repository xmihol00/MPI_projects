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
#include <string>

#include "ParallelHeatSolver.hpp"

using namespace std;
#if MEASURE_HALO_ZONE_COMPUTATION_TIME || MEASURE_COMMUNICATION_DELAY
    using namespace std::chrono;
#endif

ParallelHeatSolver::ParallelHeatSolver(const SimulationProperties &simulationProps,
                                       const MaterialProperties &materialProps)
    : HeatSolverBase(simulationProps, materialProps), 
      _simulationHyperParams{.airFlowRate = mSimulationProps.getAirflowRate(),
                             .coolerTemp = mMaterialProps.getCoolerTemperature()}
{
    MPI_Comm_size(MPI_COMM_WORLD, &_worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &_worldRank);

    mSimulationProps.getDecompGrid(_decomposition.nx, _decomposition.ny);

    // check if it is possible to create a 2D grid of nodes
    if (mMaterialProps.getEdgeSize() >= (1UL << 31) || mMaterialProps.getEdgeSize() * mMaterialProps.getEdgeSize() >= (1UL << 31))
    {
        cerr << "The edge size of the domain is too large for the MPI communication. The power of 2 of the edge size must be less than 2^31." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // compute all the later used sizes
    _sizes.globalEdge = static_cast<int>(mMaterialProps.getEdgeSize()); // MPI does not support size_t e.g. in tile allocation
    _sizes.localWidth = _sizes.globalEdge / _decomposition.nx;
    _sizes.localHeight = _sizes.globalEdge / _decomposition.ny;
    _sizes.localHeightWithHalos = _sizes.localHeight + 4;
    _sizes.localWidthWithHalos = _sizes.localWidth + 4;
    _sizes.localTile = _sizes.localHeight * _sizes.localWidth;
    _sizes.localTileWithHalos = _sizes.localWidthWithHalos * _sizes.localHeightWithHalos;
    _sizes.globalTile = _sizes.globalEdge * _sizes.globalEdge;
    _sizes.northSouthHalo = 2 * _sizes.localWidth;
    _sizes.westEastHalo = 2 * _sizes.localHeight;

    allocLocalTiles();
    initGridTopology();
    initDataTypes();

    // open output file if storing is activated
    if (!mSimulationProps.getOutputFileName().empty())
    {
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
        // 2 windows are necessary for each tile
        MPI_Win_create(_tempTiles[0].data(), _tempTiles[0].size() * sizeof(float), sizeof(float), MPI_INFO_NULL, _topologyComm, &_haloExchangeWindows[0]);
        MPI_Win_create(_tempTiles[1].data(), _tempTiles[1].size() * sizeof(float), sizeof(float), MPI_INFO_NULL, _topologyComm, &_haloExchangeWindows[1]);
    #elif RAW_EXCHANGE
        // single window is enough, the destination is always the second halo zone buffer
        MPI_Win_create(_tempHaloZones[0].data(), _tempHaloZones[0].size() * sizeof(float), sizeof(float), MPI_INFO_NULL, _topologyComm, &_haloExchangeWindows[0]);
    #endif
    }
    
    // get the neighbors of the current node
    MPI_Cart_shift(_topologyComm, 0, 1, _neighbors, _neighbors + 1);
    MPI_Cart_shift(_topologyComm, 1, 1, _neighbors + 2, _neighbors + 3);

    // counts and displacements for all to all communication with different data sizes
    _transferCounts_Raw[0] = _transferCounts_Raw[1] = _sizes.northSouthHalo;
    _transferCounts_Raw[2] = _transferCounts_Raw[3] = _sizes.westEastHalo;
    _displacements_Raw[0] = 0;
    _displacements_Raw[1] = _sizes.northSouthHalo;
    _displacements_Raw[2] = 2 * _sizes.northSouthHalo;
    _displacements_Raw[3] = 2 * _sizes.northSouthHalo + _sizes.westEastHalo;

    // necessary for RMA communication, when putting data e.g. to the north neighbor, the source of the data must be south halo zone
    _inverseDisplacements_Raw[0] = _sizes.northSouthHalo;
    _inverseDisplacements_Raw[1] = 0;
    _inverseDisplacements_Raw[2] = 2 * _sizes.northSouthHalo + _sizes.westEastHalo;
    _inverseDisplacements_Raw[3] = 2 * _sizes.northSouthHalo;
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

    // halo zones are separate buffers
    _tempHaloZones[0].resize(_sizes.localWidth * 4 + _sizes.localHeight * 4);
    _tempHaloZones[1].resize(_sizes.localWidth * 4 + _sizes.localHeight * 4);
    _domainParamsHaloZoneTmp.resize(_sizes.localWidth * 4 + _sizes.localHeight * 4);
    _domainParamsHaloZone.resize(_sizes.localWidth * 4 + _sizes.localHeight * 4);
#endif
}

void ParallelHeatSolver::initDataTypes()
{
    int outerTileSizes[2] = {_sizes.globalEdge, _sizes.globalEdge};
    int innerTileSizes[2] = {_sizes.localHeight, _sizes.localWidth};
    int starts[2] = {0, 0};
    
    // temperature/domain parameters scatter and gather data type
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_FLOAT, &_floatTileWithoutHaloZones);
    MPI_Type_commit(&_floatTileWithoutHaloZones);
    MPI_Type_create_resized(_floatTileWithoutHaloZones, 0, _sizes.localWidth * sizeof(float), &_floatTileWithoutHaloZonesResized);
    MPI_Type_commit(&_floatTileWithoutHaloZonesResized);

    // domain map scatter data type
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_INT, &_intTileWithoutHaloZones);
    MPI_Type_commit(&_intTileWithoutHaloZones);
    MPI_Type_create_resized(_intTileWithoutHaloZones, 0, _sizes.localWidth * sizeof(int), &_intTileWithoutHaloZonesResized);
    MPI_Type_commit(&_intTileWithoutHaloZonesResized);

    outerTileSizes[0] = _sizes.localHeightWithHalos;
    outerTileSizes[1] = _sizes.localWidthWithHalos;
    starts[0] = 2;
    starts[1] = 2;

    // temperature/ domain parameters local tile with halo zones data type
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_FLOAT, &_floatTileWithHaloZones);
    MPI_Type_commit(&_floatTileWithHaloZones);

    // domain map local tile with halo zones data type
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_INT, &_intTileWithHaloZones);
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

    innerTileSizes[0] = 2;
    innerTileSizes[1] = _sizes.localWidth;

    // temperature/domain parameters north halo zones for all to all exchange
    starts[0] = 2;
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_FLOAT, &_floatSendHaloZoneTypes[NORTH]);
    MPI_Type_commit(&_floatSendHaloZoneTypes[NORTH]);
    starts[0] = 0;
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_FLOAT, &_floatRecvHaloZoneTypes[NORTH]);
    MPI_Type_commit(&_floatRecvHaloZoneTypes[NORTH]);

    // temperature/domain parameters south halo zones for all to all exchange
    starts[0] = _sizes.localHeight;
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_FLOAT, &_floatSendHaloZoneTypes[SOUTH]);
    MPI_Type_commit(&_floatSendHaloZoneTypes[SOUTH]);
    starts[0] = _sizes.localHeight + 2;
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_FLOAT, &_floatRecvHaloZoneTypes[SOUTH]);
    MPI_Type_commit(&_floatRecvHaloZoneTypes[SOUTH]);

    innerTileSizes[0] = _sizes.localHeight;
    innerTileSizes[1] = 2;

    // temperature/domain parameters west halo zones for all to all exchange
    starts[0] = 2;
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_FLOAT, &_floatSendHaloZoneTypes[WEST]);
    MPI_Type_commit(&_floatSendHaloZoneTypes[WEST]);
    starts[1] = 0;
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_FLOAT, &_floatRecvHaloZoneTypes[WEST]);
    MPI_Type_commit(&_floatRecvHaloZoneTypes[WEST]);

    // temperature/domain parameters east halo zones for all to all exchange
    starts[1] = _sizes.localWidth;
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_FLOAT, &_floatSendHaloZoneTypes[EAST]);
    MPI_Type_commit(&_floatSendHaloZoneTypes[EAST]);
    starts[1] = _sizes.localWidth + 2;
    MPI_Type_create_subarray(2, outerTileSizes, innerTileSizes, starts, MPI_ORDER_C, MPI_FLOAT, &_floatRecvHaloZoneTypes[EAST]);
    MPI_Type_commit(&_floatRecvHaloZoneTypes[EAST]);

    // necessary for RMA communication, when putting data e.g. to the north neighbor, the source of the data must be south halo zone
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
    using Corners = ParallelHeatSolver::Corners;
    using CornerPoints = ParallelHeatSolver::CornerPoints;

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
    float *domainParamsTopRow0 = _domainParamsHaloZone.data();  // closest to the top
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
    float *domainParamsBotRow0 = _domainParamsHaloZone.data() + _sizes.northSouthHalo + _sizes.localWidth; // closest to the bottom

    float *domainParamsTile = _domainParamsTile.data();

    pair<float, float> *domainParamsWestHaloZone = reinterpret_cast<pair<float, float> *>(_domainParamsHaloZone.data() + 2 * _sizes.northSouthHalo);
    pair<float, float> *domainParamsEastHaloZone = reinterpret_cast<pair<float, float> *>(_domainParamsHaloZone.data() + 2 * _sizes.northSouthHalo + _sizes.westEastHalo);

    // domain map, only the center rows are needed
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
            tempTileTopRow0Next[0] = computeCornerPoint<Corners::LEFT_UPPER, CornerPoints::LEFT_UPPER>(current);

        // second most left most upper value
        tempWestHaloZoneNext[0].second = 
            tempHaloTopRow0Next[1] = 
            tempTileTopRow0Next[1] = computeCornerPoint<Corners::LEFT_UPPER, CornerPoints::RIGHT_UPPER>(current);

        // most left second most upper value
        tempWestHaloZoneNext[1].first = 
            tempHaloTopRow1Next[0] = 
            tempTileTopRow1Next[0] = computeCornerPoint<Corners::LEFT_UPPER, CornerPoints::LEFT_LOWER>(current);

        // second most left second most upper value
        tempWestHaloZoneNext[1].second = 
            tempHaloTopRow1Next[1] = 
            tempTileTopRow1Next[1] = computeCornerPoint<Corners::LEFT_UPPER, CornerPoints::RIGHT_LOWER>(current);
    }

    // north halo zone
    if (isNotTopRow()) // node is not in the top row
    {
        for (int i = 2; i < _sizes.localWidth - 2; i++)
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
            tempTileTopRow0Next[verticalIdx] = computeCornerPoint<Corners::RIGHT_UPPER, CornerPoints::LEFT_UPPER>(current);

        verticalIdx++;

        // most right most upper value
        tempEastHaloZoneNext[haloZoneIdx].second = 
            tempHaloTopRow0Next[verticalIdx] = 
            tempTileTopRow0Next[verticalIdx] = computeCornerPoint<Corners::RIGHT_UPPER, CornerPoints::RIGHT_UPPER>(current);

        verticalIdx--;
        haloZoneIdx++;

        // second most right second most upper value
        tempEastHaloZoneNext[haloZoneIdx].first = 
            tempHaloTopRow1Next[verticalIdx] = 
            tempTileTopRow1Next[verticalIdx] = computeCornerPoint<Corners::RIGHT_UPPER, CornerPoints::LEFT_LOWER>(current);

        verticalIdx++;

        // most right second most upper value
        tempEastHaloZoneNext[haloZoneIdx].second = 
            tempHaloTopRow1Next[verticalIdx] = 
            tempTileTopRow1Next[verticalIdx] = computeCornerPoint<Corners::RIGHT_UPPER, CornerPoints::RIGHT_LOWER>(current);
    }

    // west halo zone
    if (isNotLeftColumn()) // node is not in the left column
    {
        for (int i = 2; i < _sizes.localHeight - 2; i++)
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
        for (int i = 2; i < _sizes.localHeight - 2; i++)
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
            tempTileBotRow0Next[verticalIdx] = computeCornerPoint<Corners::LEFT_LOWER, CornerPoints::LEFT_LOWER>(current);

        verticalIdx++;

        // second most left most lower value
        tempWestHaloZoneNext[haloZoneIdx].second = 
            tempHaloBotRow0Next[verticalIdx] = 
            tempTileBotRow0Next[verticalIdx] = computeCornerPoint<Corners::LEFT_LOWER, CornerPoints::RIGHT_LOWER>(current);

        verticalIdx--;
        haloZoneIdx--;

        // most left second most lower value
        tempWestHaloZoneNext[haloZoneIdx].first = 
            tempHaloBotRow1Next[verticalIdx] = 
            tempTileBotRow1Next[verticalIdx] = computeCornerPoint<Corners::LEFT_LOWER, CornerPoints::LEFT_UPPER>(current);

        verticalIdx++;

        // second most left second most lower value
        tempWestHaloZoneNext[haloZoneIdx].second = 
            tempHaloBotRow1Next[verticalIdx] = 
            tempTileBotRow1Next[verticalIdx] = computeCornerPoint<Corners::LEFT_LOWER, CornerPoints::RIGHT_UPPER>(current);
    }

    // south halo zone
    if (isNotBottomRow()) // node is not in the bottom row
    {
        for (int i = 2; i < _sizes.localWidth - 2; i++)
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
            tempTileBotRow0Next[verticalIdx] = computeCornerPoint<Corners::RIGHT_LOWER, CornerPoints::LEFT_LOWER>(current);

        verticalIdx++;

        // most right most lower value
        tempEastHaloZoneNext[haloZoneIdx].second = 
            tempHaloBotRow0Next[verticalIdx] = 
            tempTileBotRow0Next[verticalIdx] = computeCornerPoint<Corners::RIGHT_LOWER, CornerPoints::RIGHT_LOWER>(current);

        verticalIdx--;
        haloZoneIdx--;

        // second most right second most lower value
        tempEastHaloZoneNext[haloZoneIdx].first = 
            tempHaloBotRow1Next[verticalIdx] = 
            tempTileBotRow1Next[verticalIdx] = computeCornerPoint<Corners::RIGHT_LOWER, CornerPoints::LEFT_UPPER>(current);

        verticalIdx++;

        // most right second most lower value
        tempEastHaloZoneNext[haloZoneIdx].second = 
            tempHaloBotRow1Next[verticalIdx] = 
            tempTileBotRow1Next[verticalIdx] = computeCornerPoint<Corners::RIGHT_LOWER, CornerPoints::RIGHT_UPPER>(current);
    }
}

void ParallelHeatSolver::computeTempHaloZones_DataType(bool current, bool next)
{
    float *tempCurrentTile = _tempTiles[current].data();
    
    float *tempCurrentTopRow0 = _tempTiles[current].data(); // closest to the top
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

    float *domainParamsTopRow0 = _domainParamsTile.data(); // closest to the top
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
    if (isNotTopRow()) // node is not in the top row
    {
        for (int i = isNotLeftColumn() ? 2 : 4; i < _sizes.localWidthWithHalos - (isNotRightColumn() ? 2 : 4); i++) // adjust offset to not compute undefined corners
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
    if (isNotLeftColumn()) // node is not in the left column
    {
        for (int i = 4; i < _sizes.localHeightWithHalos - 4; i++) // skip top and bottom halo zones
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
    if (isNotRightColumn()) // node is not in the right column
    {
        for (int i = 4; i < _sizes.localHeightWithHalos - 4; i++) // skip top and bottom halo zones
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
        for (int i = isNotLeftColumn() ? 2 : 4; i < _sizes.localWidthWithHalos - (isNotRightColumn() ? 2 : 4); i++) // adjust offset to not compute undefined corners
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
    // use raw pointers for OMP optimizations
    float *tempCurrentTile = _tempTiles[current].data();
    float *tempNextTile = _tempTiles[next].data();
    float *domainParamsTile = _domainParamsTile.data();
    int *domainMapTile = _domainMapTile.data();
    int localHeight = _sizes.localHeight;
    int localWidth = _sizes.localWidth;

    #pragma omp parallel for firstprivate(tempCurrentTile, tempNextTile, domainParamsTile, domainMapTile, localHeight, localWidth) schedule(static)
    for (int i = 2; i < localHeight - 2; i++) // offset 2 from the north and south edges (values there are already computed from the halo zones computation)
    {
        const int northUpperIdx = (i - 2) * localWidth;
        const int northIdx = (i - 1) * localWidth;
        const int centerIdx = i * localWidth;
        const int southIdx = (i + 1) * localWidth;
        const int southLowerIdx = (i + 2) * localWidth;
        const int westLeftIdx = i * localWidth - 2;
        const int westIdx = i * localWidth - 1;
        const int eastIdx = i * localWidth + 1;
        const int eastRightIdx = i * localWidth + 2;

        #pragma omp simd aligned(tempCurrentTile, tempNextTile, domainParamsTile, domainMapTile : AlignedAllocator<>::alignment) simdlen(16)
        for (int j = 2; j < localWidth - 2; j++) // offset 2 from the west and east edges (values there are already computed from the halo zones computation)
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

void ParallelHeatSolver::computeTempTile_DataType(bool current, bool next)
{
    // use raw pointers for OMP optimizations
    float *tempCurrentTile = _tempTiles[current].data();
    float *tempNextTile = _tempTiles[next].data();
    float *domainParamsTile = _domainParamsTile.data();
    int *domainMapTile = _domainMapTile.data();
    int localHeight = _sizes.localHeightWithHalos;
    int localWidth = _sizes.localWidthWithHalos;

    #pragma omp parallel for firstprivate(tempCurrentTile, tempNextTile, domainParamsTile, domainMapTile, localHeight, localWidth) schedule(static)
    for (int i = 4; i < localHeight - 4; i++) // offset 4 from the north and south edges (values there are the halo zones and already computed new values)
    {
        const int northUpperIdx = (i - 2) * localWidth;
        const int northIdx = (i - 1) * localWidth;
        const int centerIdx = i * localWidth;
        const int southIdx = (i + 1) * localWidth;
        const int southLowerIdx = (i + 2) * localWidth;
        const int westLeftIdx = centerIdx - 2;
        const int westIdx = centerIdx - 1;
        const int eastIdx = centerIdx + 1;
        const int eastRightIdx = centerIdx + 2;

        #pragma omp simd aligned(tempCurrentTile, tempNextTile, domainParamsTile, domainMapTile : AlignedAllocator<>::alignment) simdlen(16)
        for (int j = 4; j < localWidth - 4; j++) // offset 4 from the west and east edges (values there are the halo zones and already computed new values)
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
        for (int i = 0; i < _sizes.localHeight; i++)
        {
            sum += _tempTiles[!(iteration & 1)][i * _sizes.localWidth + midColIdx];
        }
        sum /= _sizes.localHeight; // average locally
    }
    else // there are multiple columns, left most column must be sampled
    {
        for (int i = 0; i < _sizes.localHeight; i++)
        {
            sum += tempWestHaloZoneNext[i].first;
        }
        sum /= _sizes.localHeight; // average locally
    }

    float reducedSum = 0;
    MPI_Reduce(&sum, &reducedSum, 1, MPI_FLOAT, MPI_SUM, 0, _midColComm);

    if (_midColRank == 0) // only the root process in the middle column communicator prints the progress report
    {
        reducedSum /= _decomposition.ny; // average globally
        printProgressReport(iteration, reducedSum);
    }
}

void ParallelHeatSolver::computeAndPrintMidColAverageParallel_DataType(size_t iteration)
{
    float sum = 0;
    bool tileIdx = !(iteration & 1);
    if (_decomposition.nx == 1) // there is only one column, middle column of the tile must be sampled (can happen only with 1 or 2 processes)
    {
        int midColIdx = _sizes.localWidthWithHalos >> 1;

        #pragma omp parallel for reduction(+ : sum) schedule(static)
        for (int i = 2; i < _sizes.localHeightWithHalos - 2; i++)
        {
            sum += _tempTiles[tileIdx][i * _sizes.localWidthWithHalos + midColIdx + 2];
        }
        sum /= _sizes.localHeight; // average locally
    }
    else // there are multiple columns, left most column must be sampled
    {
        #pragma omp parallel for reduction(+ : sum) schedule(static)
        for (int i = 2; i < _sizes.localHeightWithHalos - 2; i++)
        {
            sum += _tempTiles[tileIdx][i * _sizes.localWidthWithHalos + 1];
        }
        sum /= _sizes.localHeight; // average locally
    }

    float reducedSum = 0;
    MPI_Reduce(&sum, &reducedSum, 1, MPI_FLOAT, MPI_SUM, 0, _midColComm);

    if (_midColRank == 0) // only the root process in the middle column communicator prints the progress report
    {
        reducedSum /= _decomposition.ny; // average globally
        printProgressReport(iteration, reducedSum);
    }
}

void ParallelHeatSolver::computeAndPrintMidColAverageSequential(float timeElapsed, const vector<float, AlignedAllocator<float>> &outResult)
{
    if (_worldRank == 0) // only the root process prints the final report
    {
        float averageTemp = 0;

        #pragma omp parallel for reduction(+ : averageTemp) schedule(static)
        for (int i = 0; i < _sizes.globalEdge; i++)
        {
            averageTemp += outResult[i * _sizes.globalEdge + (_sizes.globalEdge >> 1)];
        }
        averageTemp /= _sizes.globalEdge;

        printFinalReport(timeElapsed, averageTemp);
    }
}

void ParallelHeatSolver::startHaloExchangeP2P_Raw()
{
    // leverage the created 2D topology to exchange the halo zones
    MPI_Ineighbor_alltoallv(_tempHaloZones[1].data(), _transferCounts_Raw, _displacements_Raw, MPI_FLOAT,
                            _tempHaloZones[0].data(), _transferCounts_Raw, _displacements_Raw, MPI_FLOAT, _topologyComm, 
                            &_haloExchangeRequest);
}

void ParallelHeatSolver::startHaloExchangeP2P_DataType(bool tile)
{
    // leverage the created 2D topology to exchange the halo zones
    MPI_Ineighbor_alltoallw(_tempTiles[tile].data(), _transferCounts_DataType, _displacements_DataType, _floatSendHaloZoneTypes,
                            _tempTiles[tile].data(), _transferCounts_DataType, _displacements_DataType, _floatRecvHaloZoneTypes, _topologyComm, 
                            &_haloExchangeRequest);
}

void ParallelHeatSolver::startHaloExchangeRMA_Raw()
{
    MPI_Win_fence(0, _haloExchangeWindows[0]); // synchronize the access to the window

    for (int i = 0; i < 4; i++) // north, south, west, east
    {
        if (_neighbors[i] != MPI_PROC_NULL) // if the neighbor exists (not a boundary node)
        {
            MPI_Put(_tempHaloZones[1].data() + _displacements_Raw[i], _transferCounts_Raw[i], MPI_FLOAT,
                    _neighbors[i], _inverseDisplacements_Raw[i], _transferCounts_Raw[i], MPI_FLOAT, _haloExchangeWindows[0]);
        }
    }
}

void ParallelHeatSolver::startHaloExchangeRMA_DataType(bool window_tile)
{
    MPI_Win_fence(0, _haloExchangeWindows[window_tile]); // synchronize the access to the window

    for (int i = 0; i < 4; i++) // north, south, west, east
    {
        if (_neighbors[i] != MPI_PROC_NULL) // if the neighbor exists (not a boundary node)
        {
            MPI_Put(_tempTiles[window_tile].data(), 1, _floatSendHaloZoneTypes[i],
                    _neighbors[i], 0, 1, _floatInverseRecvHaloZoneTypes[i], _haloExchangeWindows[window_tile]);
        }
    }
}

void ParallelHeatSolver::awaitHaloExchangeP2P_Raw()
{
    MPI_Status status = {0, 0, 0, 0, 0};
    MPI_Wait(&_haloExchangeRequest, &status);
    if (status.MPI_ERROR != MPI_SUCCESS) // abort in case of a failure
    {
        cerr << "Rank: " << _worldRank << " - Error in halo exchange: " << status.MPI_ERROR << endl;
        MPI_Abort(MPI_COMM_WORLD, status.MPI_ERROR);
    }
}

void ParallelHeatSolver::awaitHaloExchangeP2P_DataType(bool unused)
{
    (void)unused;
    awaitHaloExchangeP2P_Raw();
}

void ParallelHeatSolver::awaitHaloExchangeRMA_Raw()
{
    MPI_Win_fence(0, _haloExchangeWindows[0]); // synchronize the access to the window (wait for the completion of the put operations)
}

void ParallelHeatSolver::awaitHaloExchangeRMA_DataType(bool window)
{
    MPI_Win_fence(0, _haloExchangeWindows[window]); // synchronize the access to the window (wait for the completion of the put operations)
}

void ParallelHeatSolver::scatterInitialData_Raw()
{
    // scatter the temperature
    MPI_Scatterv(mMaterialProps.getInitialTemperature().data(), _scatterGatherCounts.data(), _scatterGatherDisplacements.data(), _floatTileWithoutHaloZonesResized,
                 _tempTiles[0].data(), _sizes.localTile, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // scatter the domain parameters
    MPI_Scatterv(mMaterialProps.getDomainParameters().data(), _scatterGatherCounts.data(), _scatterGatherDisplacements.data(), _floatTileWithoutHaloZonesResized,
                 _domainParamsTile.data(), _sizes.localTile, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // scatter the domain map
    MPI_Scatterv(mMaterialProps.getDomainMap().data(), _scatterGatherCounts.data(), _scatterGatherDisplacements.data(), _intTileWithoutHaloZonesResized,
                 _domainMapTile.data(), _sizes.localTile, MPI_INT, 0, MPI_COMM_WORLD);

    // copy to halo zones North and South
    copy(_tempTiles[0].begin(), _tempTiles[0].begin() + _sizes.northSouthHalo, _tempHaloZones[1].begin());                     // North
    copy(_tempTiles[0].end() - _sizes.northSouthHalo, _tempTiles[0].end(), _tempHaloZones[1].begin() + _sizes.northSouthHalo); // South

    copy(_domainParamsTile.begin(), _domainParamsTile.begin() + _sizes.northSouthHalo, _domainParamsHaloZoneTmp.begin());                     // North
    copy(_domainParamsTile.end() - _sizes.northSouthHalo, _domainParamsTile.end(), _domainParamsHaloZoneTmp.begin() + _sizes.northSouthHalo); // South

    // copy to halo zones West and East
    for (int i = 0; i < _sizes.localHeight; i++)
    {
        _tempHaloZones[1][2 * _sizes.northSouthHalo + 2 * i] = _tempTiles[0][i * _sizes.localWidth];                                                   // West
        _tempHaloZones[1][2 * _sizes.northSouthHalo + 2 * i + 1] = _tempTiles[0][i * _sizes.localWidth + 1];                                           // West
        _tempHaloZones[1][2 * _sizes.northSouthHalo + _sizes.westEastHalo + 2 * i] = _tempTiles[0][i * _sizes.localWidth + _sizes.localWidth - 2];     // East
        _tempHaloZones[1][2 * _sizes.northSouthHalo + _sizes.westEastHalo + 2 * i + 1] = _tempTiles[0][i * _sizes.localWidth + _sizes.localWidth - 1]; // East

        _domainParamsHaloZoneTmp[2 * _sizes.northSouthHalo + 2 * i] = _domainParamsTile[i * _sizes.localWidth];                                                   // West
        _domainParamsHaloZoneTmp[2 * _sizes.northSouthHalo + 2 * i + 1] = _domainParamsTile[i * _sizes.localWidth + 1];                                           // West
        _domainParamsHaloZoneTmp[2 * _sizes.northSouthHalo + _sizes.westEastHalo + 2 * i] = _domainParamsTile[i * _sizes.localWidth + _sizes.localWidth - 2];     // East
        _domainParamsHaloZoneTmp[2 * _sizes.northSouthHalo + _sizes.westEastHalo + 2 * i + 1] = _domainParamsTile[i * _sizes.localWidth + _sizes.localWidth - 1]; // East
    }

    // exchange the halo zones
    MPI_Neighbor_alltoallv(_tempHaloZones[1].data(), _transferCounts_Raw, _displacements_Raw, MPI_FLOAT,
                           _tempHaloZones[0].data(), _transferCounts_Raw, _displacements_Raw, MPI_FLOAT, _topologyComm);
    MPI_Neighbor_alltoallv(_domainParamsHaloZoneTmp.data(), _transferCounts_Raw, _displacements_Raw, MPI_FLOAT,
                           _domainParamsHaloZone.data(), _transferCounts_Raw, _displacements_Raw, MPI_FLOAT, _topologyComm);

    // copy initial temperature to the second buffer
    copy(_tempTiles[0].begin(), _tempTiles[0].end(), _tempTiles[1].begin());
}

void ParallelHeatSolver::scatterInitialData_DataType()
{
    // scatter the temperature and and exchange the initial temperature halo zones, then copy the temperature tile with halo zones to the second buffer
    MPI_Scatterv(mMaterialProps.getInitialTemperature().data(), _scatterGatherCounts.data(), _scatterGatherDisplacements.data(), _floatTileWithoutHaloZonesResized,
                 _tempTiles[0].data(), 1, _floatTileWithHaloZones, 0, MPI_COMM_WORLD);
    MPI_Neighbor_alltoallw(_tempTiles[0].data(), _transferCounts_DataType, _displacements_DataType, _floatSendHaloZoneTypes,
                           _tempTiles[0].data(), _transferCounts_DataType, _displacements_DataType, _floatRecvHaloZoneTypes, _topologyComm);
    copy(_tempTiles[0].begin(), _tempTiles[0].end(), _tempTiles[1].begin());
    
    // scatter the domain parameters and exchange the initial domain parameters halo zones
    MPI_Scatterv(mMaterialProps.getDomainParameters().data(), _scatterGatherCounts.data(), _scatterGatherDisplacements.data(), _floatTileWithoutHaloZonesResized,
                 _domainParamsTile.data(), 1, _floatTileWithHaloZones, 0, MPI_COMM_WORLD);
    MPI_Neighbor_alltoallw(_domainParamsTile.data(), _transferCounts_DataType, _displacements_DataType, _floatSendHaloZoneTypes,
                           _domainParamsTile.data(), _transferCounts_DataType, _displacements_DataType, _floatRecvHaloZoneTypes, _topologyComm);
    
    // scatter the domain map, no halo zones are required
    MPI_Scatterv(mMaterialProps.getDomainMap().data(), _scatterGatherCounts.data(), _scatterGatherDisplacements.data(), _intTileWithoutHaloZonesResized,
                 _domainMapTile.data(), 1, _intTileWithHaloZones, 0, MPI_COMM_WORLD);
}

void ParallelHeatSolver::gatherComputedTempData_Raw(bool tile, vector<float, AlignedAllocator<float>> &outResult)
{
    // gather the computed data from all nodes to the root node using non custom data type at the sending end
    MPI_Gatherv(_tempTiles[tile].data(), _sizes.localTile, MPI_FLOAT,
                outResult.data(), _scatterGatherCounts.data(), _scatterGatherDisplacements.data(), _floatTileWithoutHaloZonesResized, 0, MPI_COMM_WORLD);
}

void ParallelHeatSolver::gatherComputedTempData_DataType(bool tile, vector<float, AlignedAllocator<float>> &outResult)
{
    // gather the computed data from all nodes to the root node using custom data type at the sending end
    MPI_Gatherv(_tempTiles[tile].data(), 1, _floatTileWithHaloZones,
                outResult.data(), _scatterGatherCounts.data(), _scatterGatherDisplacements.data(), _floatTileWithoutHaloZonesResized, 0, MPI_COMM_WORLD);
}

void ParallelHeatSolver::run(vector<float, AlignedAllocator<float>> &outResult)
{
#if DATA_TYPE_EXCHANGE
    // scatter the initial data across the nodes from the root node
    scatterInitialData_DataType();
    
    // select the data exchange schema
    const auto startHaloExchangeFunction = mSimulationProps.isRunParallelRMA() ? &ParallelHeatSolver::startHaloExchangeRMA_DataType : &ParallelHeatSolver::startHaloExchangeP2P_DataType;
    const auto awaitHaloExchangeFunction = mSimulationProps.isRunParallelRMA() ? &ParallelHeatSolver::awaitHaloExchangeRMA_DataType : &ParallelHeatSolver::awaitHaloExchangeP2P_DataType;

    double startTime = MPI_Wtime();
    // run the simulation
    for (size_t iter = 0; iter < mSimulationProps.getNumIterations(); iter++)
    {
        const bool current = iter & 1;
        const bool next = !current;

        #if MEASURE_HALO_ZONE_COMPUTATION_TIME
            auto start = high_resolution_clock::now();
        #endif
        // compute temperature halo zones (and the two most outer rows and columns of the tile)
        computeTempHaloZones_DataType(current, next);
        #if MEASURE_HALO_ZONE_COMPUTATION_TIME
            auto end = high_resolution_clock::now();
            _haloZoneComputationDelay += duration_cast<nanoseconds>(end - start).count();
        #endif

        #if MEASURE_COMMUNICATION_DELAY
            auto start = high_resolution_clock::now();
        #endif
        // start the halo zone exchange (async P2P communication, or RMA communication if enabled)
        (this->*startHaloExchangeFunction)(next);
        #if MEASURE_COMMUNICATION_DELAY
            auto end = high_resolution_clock::now();
            _communicationDelay += duration_cast<nanoseconds>(end - start).count();
        #endif

        // compute the rest of the tile (inner part)
        computeTempTile_DataType(current, next);

        // wait for all halo zone exchanges to finalize
        (this->*awaitHaloExchangeFunction)(next);

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

    #if MEASURE_HALO_ZONE_COMPUTATION_TIME
        size_t localAverage = _haloZoneComputationDelay / mSimulationProps.getNumIterations();
        size_t globalAverage = 0;
        MPI_Reduce(&localAverage, &globalAverage, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (_worldRank == 0)
        {
            cout << ";" << globalAverage / _worldSize << endl;
        }
    #endif

    #if MEASURE_COMMUNICATION_DELAY
        size_t localAverage = _communicationDelay / mSimulationProps.getNumIterations();
        size_t globalAverage = 0;
        MPI_Reduce(&localAverage, &globalAverage, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (_worldRank == 0)
        {
            cout << ";" << globalAverage / _worldSize << endl;
        }
    #endif
#elif RAW_EXCHANGE
    // scatter the initial data across the nodes from the root node
    scatterInitialData_Raw();

    // deallocate no longer needed temporary buffer
    _domainParamsHaloZoneTmp.resize(0);

    // select the data exchange schema
    const auto startHaloExchangeFunction = mSimulationProps.isRunParallelRMA() ? &ParallelHeatSolver::startHaloExchangeRMA_Raw : &ParallelHeatSolver::startHaloExchangeP2P_Raw;
    const auto awaitHaloExchangeFunction = mSimulationProps.isRunParallelRMA() ? &ParallelHeatSolver::awaitHaloExchangeRMA_Raw : &ParallelHeatSolver::awaitHaloExchangeP2P_Raw;

    double startTime = MPI_Wtime();

    // run the simulation
    for (size_t iter = 0; iter < mSimulationProps.getNumIterations(); iter++)
    {
        const bool current = iter & 1;
        const bool next = !current;

        #if MEASURE_HALO_ZONE_COMPUTATION_TIME
            auto start = high_resolution_clock::now();
        #endif
        // compute temperature halo zones (and the two most outer rows and columns of the tile)
        computeTempHaloZones_Raw(current, next);
        #if MEASURE_HALO_ZONE_COMPUTATION_TIME
            auto end = high_resolution_clock::now();
            _haloZoneComputationDelay += duration_cast<nanoseconds>(end - start).count();
        #endif

        // start the halo zone exchange (async P2P communication, or RMA communication if enabled)
        (this->*startHaloExchangeFunction)();

        // compute the rest of the tile (inner part)
        computeTempTile_Raw(current, next);

        #if MEASURE_COMMUNICATION_DELAY
            auto start = high_resolution_clock::now();
        #endif
        // wait for all halo zone exchanges to finalize
        (this->*awaitHaloExchangeFunction)();
        #if MEASURE_COMMUNICATION_DELAY
            auto end = high_resolution_clock::now();
            _communicationDelay += duration_cast<nanoseconds>(end - start).count();
        #endif

        // store the data if required
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

    #if MEASURE_HALO_ZONE_COMPUTATION_TIME
        size_t localAverage = _haloZoneComputationDelay / mSimulationProps.getNumIterations();
        size_t globalAverage = 0;
        MPI_Reduce(&localAverage, &globalAverage, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (_worldRank == 0)
        {
            cout << ";" << globalAverage / _worldSize << endl;
        }
    #endif

    #if MEASURE_COMMUNICATION_DELAY
        size_t localAverage = _communicationDelay / mSimulationProps.getNumIterations();
        size_t globalAverage = 0;
        MPI_Reduce(&localAverage, &globalAverage, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (_worldRank == 0)
        {
            cout << ";" << globalAverage / _worldSize << endl;
        }
    #endif
#endif
}

bool ParallelHeatSolver::shouldComputeMiddleColumnAverageTemperature() const
{
    return _midColComm != MPI_COMM_NULL;
}

void ParallelHeatSolver::openOutputFileSequential()
{
    // Create the output file for sequential access.
    _fileHandle = H5Fcreate(mSimulationProps.getOutputFileName(codeType).c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (!_fileHandle.valid())
    {
        throw ios::failure("Cannot create output file!");
    }
}

void ParallelHeatSolver::storeDataIntoFileSequential(size_t iteration, const float *globalData)
{
    // call the base implementation
    storeDataIntoFile(_fileHandle, iteration, globalData);
}

void ParallelHeatSolver::openOutputFileParallel()
{
#ifdef H5_HAVE_PARALLEL
    // alow some optimizations by MPI with information about the written data
    MPI_Info fileSystemInfo;
    MPI_Info_create(&fileSystemInfo);
    MPI_Info_set(fileSystemInfo, "striping_factor", "1"); // number of I/O operations per process
    string strippingUnit = to_string(_sizes.localHeight * _sizes.localWidth * sizeof(float));
    MPI_Info_set(fileSystemInfo, "striping_unit", strippingUnit.c_str()); // size of the stripe in bytes

    Hdf5PropertyListHandle faplHandle(H5Pcreate(H5P_FILE_ACCESS));
    H5Pset_fapl_mpio(faplHandle, _topologyComm, fileSystemInfo);
    H5Pset_alignment(faplHandle, 1, 1 << 12); // set the alignment to 4 KB

    // use handle which will automatically release the resource
    _fileHandle = H5Fcreate(mSimulationProps.getOutputFileName(codeType).c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, faplHandle);
    MPI_Info_free(&fileSystemInfo);

    if (!_fileHandle.valid())
    {
        throw ios::failure("Cannot create output file!");
    }
#else
    throw runtime_error("Parallel HDF5 support is not available!");
#endif /* H5_HAVE_PARALLEL */
}

void ParallelHeatSolver::storeDataIntoFileParallel(size_t iteration, int halloOffset, const float *localData)
{
    if (_fileHandle == H5I_INVALID_HID)
    {
        return;
    }

#ifdef H5_HAVE_PARALLEL
    // compute the tile offsets, sizes and transfer counts
    array<hsize_t, 2> gridSizes{static_cast<hsize_t>(_sizes.globalEdge), static_cast<hsize_t>(_sizes.globalEdge)};
    array<hsize_t, 2> localGridSizes{static_cast<hsize_t>(_sizes.localHeight + 2 * halloOffset), static_cast<hsize_t>(_sizes.localWidth + 2 * halloOffset)};
    array<hsize_t, 2> localGridSizesWithoutHalo{static_cast<hsize_t>(_sizes.localHeight), static_cast<hsize_t>(_sizes.localWidth)};
    array<hsize_t, 2> tileOffsets{static_cast<hsize_t>(_sizes.localHeight * (_worldRank / _decomposition.nx)),
                                  static_cast<hsize_t>(_sizes.localWidth * (_worldRank % _decomposition.nx))};
    array<hsize_t, 2> halloOffsets{static_cast<hsize_t>(halloOffset), static_cast<hsize_t>(halloOffset)};
    array<hsize_t, 2> blockCounts{1, 1};

    // Create new HDF5 group in the output file
    string groupName = "Timestep_" + to_string(iteration / mSimulationProps.getWriteIntensity());

    Hdf5GroupHandle groupHandle(H5Gcreate(_fileHandle, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    
    {
        // Create new dataspace and dataset
        static constexpr string_view dataSetName{"Temperature"};

        Hdf5PropertyListHandle datasetPropListHandle(H5Pcreate(H5P_DATASET_CREATE));
        // set up chunking for collective write operation, 1 chunk per process corresponding to the local tile
        H5Pset_chunk(datasetPropListHandle, 2, localGridSizesWithoutHalo.data());

        Hdf5DataspaceHandle dataSpaceHandle(H5Screate_simple(2, gridSizes.data(), nullptr));
        Hdf5DatasetHandle dataSetHandle(H5Dcreate(groupHandle, dataSetName.data(), H5T_NATIVE_FLOAT, dataSpaceHandle,
                                                  H5P_DEFAULT, datasetPropListHandle, H5P_DEFAULT));
        Hdf5DataspaceHandle memSpaceHandle(H5Screate_simple(2, localGridSizes.data(), nullptr));

        // select the tile (position and size) in the global domain, i.e. the destination for the data
        H5Sselect_hyperslab(dataSpaceHandle, H5S_SELECT_SET, tileOffsets.data(), nullptr, blockCounts.data(), localGridSizesWithoutHalo.data());
        // select the tile (position and size) in the local domain with halo zones, i.e. the source of the data
        H5Sselect_hyperslab(memSpaceHandle, H5S_SELECT_SET, halloOffsets.data(), nullptr, blockCounts.data(), localGridSizesWithoutHalo.data());

        Hdf5PropertyListHandle propListHandle(H5Pcreate(H5P_DATASET_XFER));
        H5Pset_dxpl_mpio(propListHandle, H5FD_MPIO_COLLECTIVE);

        // write the data to the dataset collectively
        H5Dwrite(dataSetHandle, H5T_NATIVE_FLOAT, memSpaceHandle, dataSpaceHandle, propListHandle, localData);
    } // ensure resources are released

    {
        // 3. Store attribute with current iteration number in the group.
        static constexpr string_view attributeName{"Time"};
        Hdf5DataspaceHandle dataSpaceHandle(H5Screate(H5S_SCALAR));
        Hdf5AttributeHandle attributeHandle(H5Acreate2(groupHandle, attributeName.data(), H5T_IEEE_F64LE, dataSpaceHandle, H5P_DEFAULT, H5P_DEFAULT));
        const double snapshotTime = static_cast<double>(iteration);
        H5Awrite(attributeHandle, H5T_IEEE_F64LE, &snapshotTime);
    } // ensure resources are released
#else
    throw runtime_error("Parallel HDF5 support is not available!");
#endif /* H5_HAVE_PARALLEL */
}
