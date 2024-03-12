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

    deinitDataDistribution();
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
}

void ParallelHeatSolver::deinitGridTopology()
{
    MPI_Comm_free(&_topologyComm);
    if (_midColComm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&_midColComm);
    }
}

void ParallelHeatSolver::initDataDistribution()
{
    if (_decomposition.ny > 1)
    {
        if (_worldRank % _decomposition.nx == 0) // first columns
        {
            MPI_Comm_split(MPI_COMM_WORLD, 0, _worldRank / _decomposition.nx, &_scatterGatherColComm);
            MPI_Comm_set_name(_scatterGatherColComm, "Initial Scatter Column Communicator");
        }
        else // other columns
        {
            MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, MPI_UNDEFINED, &_scatterGatherColComm);
            _scatterGatherColComm = MPI_COMM_NULL;
        }
    }
    else
    {
        _scatterGatherColComm = MPI_COMM_NULL;
    }

    MPI_Comm_split(MPI_COMM_WORLD, _worldRank / _decomposition.nx, _worldRank % _decomposition.nx, &_scatterGatherRowComm);
    MPI_Comm_rank(_scatterGatherRowComm, &_rowRank);
    MPI_Comm_set_name(_scatterGatherRowComm, "Initial Scatter Row Communicator");
}

void ParallelHeatSolver::deinitDataDistribution()
{
    if (_decomposition.ny > 1 && _scatterGatherColComm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&_scatterGatherColComm);
    }
    MPI_Comm_free(&_scatterGatherRowComm);
}

void ParallelHeatSolver::allocLocalTiles()
{
    _tempTiles[0].resize(_edgeSizes.localWidth * _edgeSizes.localHeight);
    _tempTiles[1].resize(_edgeSizes.localWidth * _edgeSizes.localHeight);
    _domainParamsTile.resize(_edgeSizes.localWidth * _edgeSizes.localHeight);
    _domainMapTile.resize(_edgeSizes.localWidth * _edgeSizes.localHeight);

    _tempHaloZones[0].resize(_edgeSizes.localWidth * 4 + _edgeSizes.localHeight * 4);
    _tempHaloZones[1].resize(_edgeSizes.localWidth * 4 + _edgeSizes.localHeight * 4);
    _domainParamsHaloZoneTmp.resize(_edgeSizes.localWidth * 4 + _edgeSizes.localHeight * 4);
    _domainParamsHaloZone.resize(_edgeSizes.localWidth * 4 + _edgeSizes.localHeight * 4);

    if (_scatterGatherColComm != MPI_COMM_NULL)
    {
        _scatterGatherTempRow.resize(_edgeSizes.global * _edgeSizes.localHeight);
        _initialScatterDomainParams.resize(_edgeSizes.global * _edgeSizes.localHeight);
        _initialScatterDomainMap.resize(_edgeSizes.global * _edgeSizes.localHeight);
    }
}

void ParallelHeatSolver::computeTempHaloZones(bool current, bool next)
{
    enum class Corners { LEFT_UPPER, RIGHT_UPPER, LEFT_LOWER, RIGHT_LOWER };
    enum class CornerPoints { LEFT_UPPER, RIGHT_UPPER, LEFT_LOWER, RIGHT_LOWER };
    auto computeCornerPoint = [&]<Corners corner, CornerPoints cornerPoint>() 
    {   
        pair<float, float> *tempWestHaloZone = reinterpret_cast<pair<float, float> *>(_tempHaloZones[0].data() + 2 * _offsets.northSouthHalo);
        pair<float, float> *tempEastHaloZone = reinterpret_cast<pair<float, float> *>(_tempHaloZones[0].data() + 2 * _offsets.northSouthHalo + _offsets.westEastHalo);
        
        pair<float, float> *domainParamsWestHaloZone = reinterpret_cast<pair<float, float> *>(_domainParamsHaloZone.data() + 2 * _offsets.northSouthHalo);
        pair<float, float> *domainParamsEastHaloZone = reinterpret_cast<pair<float, float> *>(_domainParamsHaloZone.data() + 2 * _offsets.northSouthHalo + _offsets.westEastHalo);

        float tempNorthUpper, tempNorth, tempSouth, tempSouthLower, tempWestLeft, tempWest, tempEast, tempEastRight, tempCenter;
        float domainParamsNorthUpper, domainParamsNorth, domainParamsSouth, domainParamsSouthLower, domainParamsWestLeft, domainParamsWest, domainParamsEast, domainParamsEastRight, domainParamsCenter;
        int domainMapCenter;
        if constexpr (corner == Corners::LEFT_UPPER || corner == Corners::RIGHT_UPPER)
        {          
            size_t rowCornerOffset = corner == Corners::RIGHT_UPPER ? _edgeSizes.localWidth - 2 : 0;
            constexpr size_t haloZoneOffset = cornerPoint == CornerPoints::LEFT_UPPER || cornerPoint == CornerPoints::RIGHT_UPPER ? 0 : 1;
            size_t rowOffset = cornerPoint == CornerPoints::LEFT_UPPER ? rowCornerOffset :
                                   cornerPoint == CornerPoints::RIGHT_UPPER ? rowCornerOffset + 1 :
                                       cornerPoint == CornerPoints::LEFT_LOWER ? rowCornerOffset + _edgeSizes.localWidth :
                                           rowCornerOffset + _edgeSizes.localWidth + 1;

            if constexpr (cornerPoint == CornerPoints::LEFT_UPPER || cornerPoint == CornerPoints::RIGHT_UPPER)
            {
                tempNorthUpper = _tempHaloZones[0][rowOffset];
                tempNorth = _tempHaloZones[0][_edgeSizes.localWidth + rowOffset];
                tempCenter = _tempTiles[current][rowOffset];
                tempSouth = _tempTiles[current][_edgeSizes.localWidth + rowOffset];
                tempSouthLower = _tempTiles[current][2 * _edgeSizes.localWidth + rowOffset];

                domainParamsNorthUpper = _domainParamsHaloZone[rowOffset];
                domainParamsNorth = _domainParamsHaloZone[_edgeSizes.localWidth + rowOffset];
                domainParamsCenter = _domainParamsTile[rowOffset];
                domainParamsSouth = _domainParamsTile[_edgeSizes.localWidth + rowOffset];
                domainParamsSouthLower = _domainParamsTile[2 * _edgeSizes.localWidth + rowOffset];

                domainMapCenter = _domainMapTile[rowOffset];
            }
            else
            {
                tempNorthUpper = _tempHaloZones[0][rowOffset];
                tempNorth = _tempTiles[current][rowOffset - _edgeSizes.localWidth];
                tempCenter = _tempTiles[current][rowOffset];
                tempSouth = _tempTiles[current][_edgeSizes.localWidth + rowOffset];
                tempSouthLower = _tempTiles[current][2 * _edgeSizes.localWidth + rowOffset];

                domainParamsNorthUpper = _domainParamsHaloZone[rowOffset];
                domainParamsNorth = _domainParamsTile[rowOffset - _edgeSizes.localWidth];
                domainParamsCenter = _domainParamsTile[rowOffset];
                domainParamsSouth = _domainParamsTile[_edgeSizes.localWidth + rowOffset];
                domainParamsSouthLower = _domainParamsTile[2 * _edgeSizes.localWidth + rowOffset];

                domainMapCenter = _domainMapTile[rowOffset];
            }

            if constexpr (corner == Corners::LEFT_UPPER)
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
        }
        else
        {
            if constexpr (cornerPoint == CornerPoints::LEFT_UPPER || cornerPoint == CornerPoints::RIGHT_UPPER)
            {

            }
            else
            {

            }

            if constexpr (cornerPoint == CornerPoints::RIGHT_UPPER || cornerPoint == CornerPoints::RIGHT_LOWER)
            {

            }
            else
            {

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
    float *tempTopRow1Current = _tempHaloZones[0].data() + _edgeSizes.localWidth;
    float *tempTopRow2Current = _tempTiles[current].data();
    float *tempTopRow3Current = _tempTiles[current].data() + _edgeSizes.localWidth;
    float *tempTopRow4Current = _tempTiles[current].data() + _offsets.northSouthHalo;
    float *tempTopRow5Current = _tempTiles[current].data() + _offsets.northSouthHalo + _edgeSizes.localWidth;

    float *tempBotRow5Current = _tempTiles[current].data() + _edgeSizes.localHeight * _edgeSizes.localWidth - 2 * _offsets.northSouthHalo;
    float *tempBotRow4Current = _tempTiles[current].data() + _edgeSizes.localHeight * _edgeSizes.localWidth - _offsets.northSouthHalo - _edgeSizes.localWidth;
    float *tempBotRow3Current = _tempTiles[current].data() + _edgeSizes.localHeight * _edgeSizes.localWidth - _offsets.northSouthHalo;
    float *tempBotRow2Current = _tempTiles[current].data() + _edgeSizes.localHeight * _edgeSizes.localWidth - _edgeSizes.localWidth;
    float *tempBotRow1Current = _tempHaloZones[0].data() + _offsets.northSouthHalo;
    float *tempBotRow0Current = _tempHaloZones[0].data() + _offsets.northSouthHalo + _edgeSizes.localWidth; // closest to the bottom

    // cast to pairs for easier access
    pair<float, float> *tempWestHaloZoneCurrent = reinterpret_cast<pair<float, float> *>(_tempHaloZones[0].data() + 2 * _offsets.northSouthHalo);
    pair<float, float> *tempEastHaloZoneCurrent = reinterpret_cast<pair<float, float> *>(_tempHaloZones[0].data() + 2 * _offsets.northSouthHalo + _offsets.westEastHalo);

    // rows to store results to
    float *tempHaloTopRow0Next = _tempHaloZones[1].data(); // closest to the top
    float *tempHaloTopRow1Next = _tempHaloZones[1].data() + _edgeSizes.localWidth;
    float *tempTileTopRow0Next = _tempTiles[next].data(); // closest to the top
    float *tempTileTopRow1Next = _tempTiles[next].data() + _edgeSizes.localWidth;

    float *tempHaloBotRow1Next = _tempHaloZones[1].data() + _offsets.northSouthHalo;
    float *tempHaloBotRow0Next = _tempHaloZones[1].data() + _offsets.northSouthHalo + _edgeSizes.localWidth; // closest to the bottom
    float *tempTileBotRow1Next = _tempTiles[next].data() + _edgeSizes.localHeight * _edgeSizes.localWidth - _offsets.northSouthHalo;
    float *tempTileBotRow0Next = _tempTiles[next].data() + _edgeSizes.localHeight * _edgeSizes.localWidth - _edgeSizes.localWidth; // closest to the bottom

    // columns to store results to (only halo zones)
    pair<float, float> *tempWestHaloZoneNext = reinterpret_cast<pair<float, float> *>(_tempHaloZones[1].data() + 2 * _offsets.northSouthHalo);
    pair<float, float> *tempEastHaloZoneNext = reinterpret_cast<pair<float, float> *>(_tempHaloZones[1].data() + 2 * _offsets.northSouthHalo + _offsets.westEastHalo);

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

    float *domainParamsTile = _domainParamsTile.data();

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
    if (!isTopRow() && !isRightColumn()) // node is not in the top row and not in the right column
    {
        // the corner consists of 4 values, which must be stored to the tile, north and east halo zones
        int verticalIdx = _edgeSizes.localWidth - 2;
        int haloZoneIdx = 0;
        int firstHorizontalIdx = _edgeSizes.localWidth - 4;
        int secondHorizontalIdx = _edgeSizes.localWidth - 3;
        int thirdHorizontalIdx = _edgeSizes.localWidth - 1;

        // second most right most upper value
        tempEastHaloZoneNext[haloZoneIdx].first = 
            tempHaloTopRow0Next[verticalIdx] = 
            tempTileTopRow0Next[verticalIdx] = computeCornerPoint.operator()<Corners::RIGHT_UPPER, CornerPoints::LEFT_UPPER>();

        verticalIdx = _edgeSizes.localWidth - 1;
        firstHorizontalIdx = _edgeSizes.localWidth - 3;
        secondHorizontalIdx = _edgeSizes.localWidth - 2;

        // most right most upper value
        tempEastHaloZoneNext[haloZoneIdx].second = 
            tempHaloTopRow0Next[verticalIdx] = 
            tempTileTopRow0Next[verticalIdx] = computeCornerPoint.operator()<Corners::RIGHT_UPPER, CornerPoints::RIGHT_UPPER>();

        verticalIdx = _edgeSizes.localWidth - 2;
        haloZoneIdx = 1;
        firstHorizontalIdx = _edgeSizes.localWidth - 4;
        secondHorizontalIdx = _edgeSizes.localWidth - 3;

        // second most right second most upper value
        tempEastHaloZoneNext[haloZoneIdx].first = 
            tempHaloTopRow1Next[verticalIdx] = 
            tempTileTopRow1Next[verticalIdx] = computeCornerPoint.operator()<Corners::RIGHT_UPPER, CornerPoints::LEFT_LOWER>();

        verticalIdx = _edgeSizes.localWidth - 1;
        firstHorizontalIdx = _edgeSizes.localWidth - 3;
        secondHorizontalIdx = _edgeSizes.localWidth - 2;

        // most right second most upper value
        tempEastHaloZoneNext[haloZoneIdx].second = 
            tempHaloTopRow1Next[verticalIdx] = 
            tempTileTopRow1Next[verticalIdx] = computeCornerPoint.operator()<Corners::RIGHT_UPPER, CornerPoints::RIGHT_LOWER>();
    }

    // columns
    if (!isLeftColumn()) // node is not in the left column
    {
        for (size_t i = 2; i < _edgeSizes.localHeight - 2; i++)
        {
            int northUpperIdx = (i - 2) * _edgeSizes.localWidth;
            int northIdx = (i - 1) * _edgeSizes.localWidth;
            int centerIdx = i * _edgeSizes.localWidth;
            int southIdx = (i + 1) * _edgeSizes.localWidth;
            int southLowerIdx = (i + 2) * _edgeSizes.localWidth;
            int eastIdx = i * _edgeSizes.localWidth + 1;
            int eastRightIdx = i * _edgeSizes.localWidth + 2;
            int westIdx = i * _edgeSizes.localWidth;

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

    if (!isRightColumn()) // node is not in the right column
    {
        for (size_t i = 2; i < _edgeSizes.localHeight - 2; i++)
        {
            int northUpperIdx = (i - 2) * _edgeSizes.localWidth + _edgeSizes.localWidth - 2;
            int northIdx = (i - 1) * _edgeSizes.localWidth + _edgeSizes.localWidth - 2;
            int centerIdx = i * _edgeSizes.localWidth + _edgeSizes.localWidth - 2;
            int southIdx = (i + 1) * _edgeSizes.localWidth + _edgeSizes.localWidth - 2;
            int southLowerIdx = (i + 2) * _edgeSizes.localWidth + _edgeSizes.localWidth - 2;
            int westLeftIdx = i * _edgeSizes.localWidth + _edgeSizes.localWidth - 4;
            int westIdx = i * _edgeSizes.localWidth + _edgeSizes.localWidth - 3;
            int eastIdx = i * _edgeSizes.localWidth + _edgeSizes.localWidth - 1;

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
    if (!isBottomRow() && !isLeftColumn()) // node is not in the bottom row and not in the left column
    {
        // the corner consists of 4 values, which must be stored to the tile, south and west halo zones
        int verticalIdx = 0;
        int haloZoneIdx = _edgeSizes.localHeight - 1;
        int firstHorizontalIdx = 1;
        int secondHorizontalIdx = 2;
        int thirdHorizontalIdx = 3;

        // most left most lower value
        tempWestHaloZoneNext[haloZoneIdx].first = tempHaloBotRow0Next[verticalIdx] = tempTileBotRow0Next[verticalIdx] = computePoint(
            tempBotRow4Current[verticalIdx], tempBotRow3Current[verticalIdx], tempBotRow1Current[verticalIdx], tempBotRow0Current[verticalIdx],
            tempWestHaloZoneCurrent[haloZoneIdx].first, tempWestHaloZoneCurrent[haloZoneIdx].second, tempBotRow2Current[1], tempBotRow2Current[2],
            tempBotRow2Current[verticalIdx],
            domainParamsBotRow4[verticalIdx], domainParamsBotRow3[verticalIdx], domainParamsBotRow1[verticalIdx], domainParamsBotRow0[verticalIdx],
            domainParamsWestHaloZone[haloZoneIdx].first, domainParamsWestHaloZone[haloZoneIdx].second, domainParamsBotRow2[1], domainParamsBotRow2[2],
            domainParamsBotRow2[verticalIdx],
            domainMapBotCenter0[verticalIdx]);

        verticalIdx = 1;
        firstHorizontalIdx = 0;

        // second most left most lower value
        tempWestHaloZoneNext[haloZoneIdx].second = tempHaloBotRow0Next[verticalIdx] = tempTileBotRow0Next[verticalIdx] = computePoint(
            tempBotRow4Current[verticalIdx], tempBotRow3Current[verticalIdx], tempBotRow1Current[verticalIdx], tempBotRow0Current[verticalIdx],
            tempWestHaloZoneCurrent[haloZoneIdx].second, tempBotRow2Current[firstHorizontalIdx], tempBotRow2Current[secondHorizontalIdx], tempBotRow2Current[thirdHorizontalIdx],
            tempBotRow2Current[verticalIdx],
            domainParamsBotRow4[verticalIdx], domainParamsBotRow3[verticalIdx], domainParamsBotRow1[verticalIdx], domainParamsBotRow0[verticalIdx],
            domainParamsWestHaloZone[haloZoneIdx].second, domainParamsBotRow2[firstHorizontalIdx], domainParamsBotRow2[secondHorizontalIdx], domainParamsBotRow2[thirdHorizontalIdx],
            domainParamsBotRow2[verticalIdx],
            domainMapBotCenter0[verticalIdx]);

        verticalIdx = 0;
        haloZoneIdx = _edgeSizes.localHeight - 2;
        firstHorizontalIdx = 1;

        // most left second most lower value
        tempWestHaloZoneNext[haloZoneIdx].first = tempHaloBotRow1Next[verticalIdx] = tempTileBotRow1Next[verticalIdx] = computePoint(
            tempBotRow5Current[verticalIdx], tempBotRow4Current[verticalIdx], tempBotRow2Current[verticalIdx], tempBotRow1Current[verticalIdx],
            tempWestHaloZoneCurrent[haloZoneIdx].first, tempWestHaloZoneCurrent[haloZoneIdx].second, tempBotRow3Current[firstHorizontalIdx], tempBotRow3Current[secondHorizontalIdx],
            tempBotRow3Current[verticalIdx],
            domainParamsBotRow5[verticalIdx], domainParamsBotRow4[verticalIdx], domainParamsBotRow2[verticalIdx], domainParamsBotRow1[verticalIdx],
            domainParamsWestHaloZone[haloZoneIdx].first, domainParamsWestHaloZone[haloZoneIdx].second, domainParamsBotRow3[firstHorizontalIdx], domainParamsBotRow3[secondHorizontalIdx],
            domainParamsBotRow3[verticalIdx],
            domainMapBotCenter1[verticalIdx]);

        verticalIdx = 1;
        firstHorizontalIdx = 0;

        // second most left second most lower value
        tempWestHaloZoneNext[haloZoneIdx].second = tempHaloBotRow1Next[verticalIdx] = tempTileBotRow1Next[verticalIdx] = computePoint(
            tempBotRow5Current[verticalIdx], tempBotRow4Current[verticalIdx], tempBotRow2Current[verticalIdx], tempBotRow1Current[verticalIdx],
            tempWestHaloZoneCurrent[haloZoneIdx].second, tempBotRow3Current[firstHorizontalIdx], tempBotRow3Current[secondHorizontalIdx], tempBotRow3Current[thirdHorizontalIdx],
            tempBotRow3Current[verticalIdx],
            domainParamsBotRow5[verticalIdx], domainParamsBotRow4[verticalIdx], domainParamsBotRow2[verticalIdx], domainParamsBotRow1[verticalIdx],
            domainParamsWestHaloZone[haloZoneIdx].second, domainParamsBotRow3[firstHorizontalIdx], domainParamsBotRow3[secondHorizontalIdx], domainParamsBotRow3[thirdHorizontalIdx],
            domainParamsBotRow3[verticalIdx],
            domainMapBotCenter1[verticalIdx]);
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
    if (!isBottomRow() && !isRightColumn()) // node is not in the bottom row and not in the right column
    {
        // the corner consists of 4 values, which must be stored to the tile, south and east halo zones
        int verticalIdx = _edgeSizes.localWidth - 2;
        int haloZoneIdx = _edgeSizes.localHeight - 1;
        int firstHorizontalIdx = _edgeSizes.localWidth - 4;
        int secondHorizontalIdx = _edgeSizes.localWidth - 3;
        int thirdHorizontalIdx = _edgeSizes.localWidth - 1;

        // second most right most lower value
        tempEastHaloZoneNext[haloZoneIdx].first = tempHaloBotRow0Next[verticalIdx] = tempTileBotRow0Next[verticalIdx] = computePoint(
            tempBotRow4Current[verticalIdx], tempBotRow3Current[verticalIdx], tempBotRow1Current[verticalIdx], tempBotRow0Current[verticalIdx],
            tempBotRow2Current[firstHorizontalIdx], tempBotRow2Current[secondHorizontalIdx], tempBotRow2Current[thirdHorizontalIdx], tempEastHaloZoneCurrent[haloZoneIdx].first,
            tempBotRow2Current[verticalIdx],
            domainParamsBotRow4[verticalIdx], domainParamsBotRow3[verticalIdx], domainParamsBotRow1[verticalIdx], domainParamsBotRow0[verticalIdx],
            domainParamsBotRow2[firstHorizontalIdx], domainParamsBotRow2[secondHorizontalIdx], domainParamsBotRow2[thirdHorizontalIdx], domainParamsEastHaloZone[haloZoneIdx].first,
            domainParamsBotRow2[verticalIdx],
            domainMapBotCenter0[verticalIdx]);

        verticalIdx = _edgeSizes.localWidth - 1;
        firstHorizontalIdx = _edgeSizes.localWidth - 3;
        secondHorizontalIdx = _edgeSizes.localWidth - 2;

        // most right most lower value
        tempEastHaloZoneNext[haloZoneIdx].second = tempHaloBotRow0Next[verticalIdx] = tempTileBotRow0Next[verticalIdx] = computePoint(
            tempBotRow4Current[verticalIdx], tempBotRow3Current[verticalIdx], tempBotRow1Current[verticalIdx], tempBotRow0Current[verticalIdx],
            tempBotRow2Current[firstHorizontalIdx], tempBotRow2Current[secondHorizontalIdx], tempEastHaloZoneCurrent[haloZoneIdx].first, tempEastHaloZoneCurrent[haloZoneIdx].second,
            tempBotRow2Current[verticalIdx],
            domainParamsBotRow4[verticalIdx], domainParamsBotRow3[verticalIdx], domainParamsBotRow1[verticalIdx], domainParamsBotRow0[verticalIdx],
            domainParamsBotRow2[firstHorizontalIdx], domainParamsBotRow2[secondHorizontalIdx], domainParamsEastHaloZone[haloZoneIdx].first, domainParamsEastHaloZone[haloZoneIdx].second,
            domainParamsBotRow2[verticalIdx],
            domainMapBotCenter0[verticalIdx]);

        verticalIdx = _edgeSizes.localWidth - 2;
        haloZoneIdx = _edgeSizes.localHeight - 2;
        firstHorizontalIdx = _edgeSizes.localWidth - 4;
        secondHorizontalIdx = _edgeSizes.localWidth - 3;

        // second most right second most lower value
        tempEastHaloZoneNext[haloZoneIdx].first = tempHaloBotRow1Next[verticalIdx] = tempTileBotRow1Next[verticalIdx] = computePoint(
            tempBotRow5Current[verticalIdx], tempBotRow4Current[verticalIdx], tempBotRow2Current[verticalIdx], tempBotRow1Current[verticalIdx],
            tempBotRow3Current[firstHorizontalIdx], tempBotRow3Current[secondHorizontalIdx], tempBotRow3Current[thirdHorizontalIdx], tempEastHaloZoneCurrent[haloZoneIdx].first,
            tempBotRow3Current[verticalIdx],
            domainParamsBotRow5[verticalIdx], domainParamsBotRow4[verticalIdx], domainParamsBotRow2[verticalIdx], domainParamsBotRow1[verticalIdx],
            domainParamsBotRow3[firstHorizontalIdx], domainParamsBotRow3[secondHorizontalIdx], domainParamsBotRow3[thirdHorizontalIdx], domainParamsEastHaloZone[haloZoneIdx].first,
            domainParamsBotRow3[verticalIdx],
            domainMapBotCenter1[verticalIdx]);

        verticalIdx = _edgeSizes.localWidth - 1;
        firstHorizontalIdx = _edgeSizes.localWidth - 3;
        secondHorizontalIdx = _edgeSizes.localWidth - 2;

        // most right second most lower value
        tempEastHaloZoneNext[haloZoneIdx].second = tempHaloBotRow1Next[verticalIdx] = tempTileBotRow1Next[verticalIdx] = computePoint(
            tempBotRow5Current[verticalIdx], tempBotRow4Current[verticalIdx], tempBotRow2Current[verticalIdx], tempBotRow1Current[verticalIdx],
            tempBotRow3Current[firstHorizontalIdx], tempBotRow3Current[secondHorizontalIdx], tempEastHaloZoneCurrent[haloZoneIdx].first, tempEastHaloZoneCurrent[haloZoneIdx].second,
            tempBotRow3Current[verticalIdx],
            domainParamsBotRow5[verticalIdx], domainParamsBotRow4[verticalIdx], domainParamsBotRow2[verticalIdx], domainParamsBotRow1[verticalIdx],
            domainParamsBotRow3[firstHorizontalIdx], domainParamsBotRow3[secondHorizontalIdx], domainParamsEastHaloZone[haloZoneIdx].first, domainParamsEastHaloZone[haloZoneIdx].second,
            domainParamsBotRow3[verticalIdx],
            domainMapBotCenter1[verticalIdx]);
    }
}

void ParallelHeatSolver::computeTempTile(bool current, bool next)
{
    float *tempCurrentTile = _tempTiles[current].data();
    float *tempNextTile = _tempTiles[next].data();

    for (size_t i = 2; i < _edgeSizes.localHeight - 2; i++)
    {
        const int northUpperIdx = (i - 2) * _edgeSizes.localWidth;
        const int northIdx = (i - 1) * _edgeSizes.localWidth;
        const int centerIdx = i * _edgeSizes.localWidth;
        const int southIdx = (i + 1) * _edgeSizes.localWidth;
        const int southLowerIdx = (i + 2) * _edgeSizes.localWidth;
        const int westLeftIdx = i * _edgeSizes.localWidth - 2;
        const int westIdx = i * _edgeSizes.localWidth - 1;
        const int eastIdx = i * _edgeSizes.localWidth + 1;
        const int eastRightIdx = i * _edgeSizes.localWidth + 2;

        for (size_t j = 2; j < _edgeSizes.localWidth - 2; j++)
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

void ParallelHeatSolver::computeAndPrintMidColAverageParallel(size_t iteration)
{
    float sum = 0;
    pair<float, float> *tempWestHaloZoneNext = reinterpret_cast<pair<float, float> *>(_tempHaloZones[1].data() + 2 * _offsets.northSouthHalo);
    if (_decomposition.nx == 1) // there is only one column, middle column of the tile must be sampled (can happen only with 1 or 2 processes)
    {
        int midColIdx = _edgeSizes.localWidth >> 1;
        for (size_t i = 0; i < _edgeSizes.localHeight; i++)
        {
            sum += _tempTiles[1][i * _edgeSizes.localWidth + midColIdx];
        }
        sum /= _edgeSizes.localHeight;
    }
    else // there are multiple columns, left most column must be sampled
    {
        for (size_t i = 0; i < _edgeSizes.localHeight; i++)
        {
            sum += tempWestHaloZoneNext[i].first;
        }
        sum /= _edgeSizes.localHeight;
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

void ParallelHeatSolver::startHaloExchangeP2P()
{
    // leverage the created 2D topology to exchange the halo zones
    MPI_Ineighbor_alltoallv(_tempHaloZones[1].data(), _transferCounts, _displacements, MPI_FLOAT,
                            _tempHaloZones[0].data(), _transferCounts, _displacements, MPI_FLOAT, _topologyComm, &_haloExchangeRequest);
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

    // int actualRecvCount;
    // MPI_Get_count(&status, MPI_FLOAT, &actualRecvCount);
    // if (actualRecvCount != _offsets.northSouthHalo * 2 + _offsets.westEastHalo * 2)
    //{
    //     cerr << "Rank: " << _worldRank << " - Error in halo exchange: received unexpected number of values. Expected: " <<  _offsets.northSouthHalo * 2 + _offsets.westEastHalo * 2  << " Received: " << actualRecvCount << endl;
    //     MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COUNT);
    // }
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
    if (_scatterGatherColComm != MPI_COMM_NULL)
    {
        auto scatter = [&]<class T>(const T *data, T *buffer)
        {
            MPI_Scatter(data, _edgeSizes.global * _edgeSizes.localHeight, is_same<T, float>::value ? MPI_FLOAT : MPI_INT,
                        buffer, _edgeSizes.global * _edgeSizes.localHeight, MPI_FLOAT, 0, _scatterGatherColComm);
        };

        scatter(scatteredTempRow, _scatterGatherTempRow.data());
        scatteredTempRow = _scatterGatherTempRow.data();

        scatter(scatteredDomainParamsRow, _initialScatterDomainParams.data());
        scatteredDomainParamsRow = _initialScatterDomainParams.data();

        scatter(scatteredDomainMapRow, _initialScatterDomainMap.data());
        scatteredDomainMapRow = _initialScatterDomainMap.data();
    }

    // scatter the initial data across each row of nodes from its first column node
    for (size_t i = 0, j = 0; i < _edgeSizes.localHeight * _edgeSizes.global; i += _edgeSizes.global, j += _edgeSizes.localWidth)
    {
        auto scatter = [&]<class T>(const T *data, T *buffer)
        {
            MPI_Scatter(data + i, _edgeSizes.localWidth, is_same<T, float>::value ? MPI_FLOAT : MPI_INT,
                        buffer + j, _edgeSizes.localWidth, MPI_FLOAT, 0, _scatterGatherRowComm);
        };

        scatter(scatteredTempRow, _tempTiles[0].data());
        scatter(scatteredDomainParamsRow, _domainParamsTile.data());
        scatter(scatteredDomainMapRow, _domainMapTile.data());
    }

    // copy initial temperature to the second buffer
    copy(_tempTiles[0].begin(), _tempTiles[0].end(), _tempTiles[1].begin());
}

void ParallelHeatSolver::gatherComputedTempData(bool final, vector<float, AlignedAllocator<float>> &outResult)
{
    if (_worldRank == 0)
    {
        outResult.resize(_edgeSizes.global * _edgeSizes.global);
    }

    float *rowTemp = outResult.data();
    if (_scatterGatherColComm != MPI_COMM_NULL)
    {
        rowTemp = _scatterGatherTempRow.data();
    }

    // gather the computed data from each row of nodes to the first column node
    for (size_t i = 0, j = 0; i < _edgeSizes.localHeight * _edgeSizes.global; i += _edgeSizes.global, j += _edgeSizes.localWidth)
    {
        MPI_Gather(_tempTiles[final].data() + j, _edgeSizes.localWidth, MPI_FLOAT,
                   rowTemp + i, _edgeSizes.localWidth, MPI_FLOAT, 0, _scatterGatherRowComm);
    }

    // gather the computed data from each node of the first column to the first node
    if (_scatterGatherColComm != MPI_COMM_NULL)
    {
        MPI_Gather(_scatterGatherTempRow.data(), _edgeSizes.global * _edgeSizes.localHeight, MPI_FLOAT,
                   outResult.data(), _edgeSizes.global * _edgeSizes.localHeight, MPI_FLOAT, 0, _scatterGatherColComm);
    }
}

void ParallelHeatSolver::prepareInitialHaloZones()
{
    // copy to halo zones North and South
    // use the odd halo zone, so that after the initial exchange, the even halo zone can be used for the first iteration
    copy(_tempTiles[0].begin(), _tempTiles[0].begin() + _offsets.northSouthHalo, _tempHaloZones[1].begin());                       // North
    copy(_tempTiles[0].end() - _offsets.northSouthHalo, _tempTiles[0].end(), _tempHaloZones[1].begin() + _offsets.northSouthHalo); // South

    copy(_domainParamsTile.begin(), _domainParamsTile.begin() + _offsets.northSouthHalo, _domainParamsHaloZoneTmp.begin());                       // North
    copy(_domainParamsTile.end() - _offsets.northSouthHalo, _domainParamsTile.end(), _domainParamsHaloZoneTmp.begin() + _offsets.northSouthHalo); // South

    // copy to halo zones West and East
    for (size_t i = 0; i < _edgeSizes.localHeight; i++)
    {
        // use the odd halo zone, so that after the initial exchange, the even halo zone can be used for the first iteration
        _tempHaloZones[1][2 * _offsets.northSouthHalo + 2 * i] = _tempTiles[0][i * _edgeSizes.localWidth];                                                         // West
        _tempHaloZones[1][2 * _offsets.northSouthHalo + 2 * i + 1] = _tempTiles[0][i * _edgeSizes.localWidth + 1];                                                 // West
        _tempHaloZones[1][2 * _offsets.northSouthHalo + _offsets.westEastHalo + 2 * i] = _tempTiles[0][i * _edgeSizes.localWidth + _edgeSizes.localWidth - 2];     // East
        _tempHaloZones[1][2 * _offsets.northSouthHalo + _offsets.westEastHalo + 2 * i + 1] = _tempTiles[0][i * _edgeSizes.localWidth + _edgeSizes.localWidth - 1]; // East

        _domainParamsHaloZoneTmp[2 * _offsets.northSouthHalo + 2 * i] = _domainParamsTile[i * _edgeSizes.localWidth];                                                         // West
        _domainParamsHaloZoneTmp[2 * _offsets.northSouthHalo + 2 * i + 1] = _domainParamsTile[i * _edgeSizes.localWidth + 1];                                                 // West
        _domainParamsHaloZoneTmp[2 * _offsets.northSouthHalo + _offsets.westEastHalo + 2 * i] = _domainParamsTile[i * _edgeSizes.localWidth + _edgeSizes.localWidth - 2];     // East
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

void ParallelHeatSolver::run(vector<float, AlignedAllocator<float>> &outResult)
{
    // scatter the initial data across the nodes from the root node
    scatterInitialData();

    // prepare the halo zone, i.e. copy the initial data to the halo zones
    prepareInitialHaloZones();

    // perform the initial halo zone exchange (synchronous P2P communication)
    exchangeInitialHaloZones();

    // deallocate no longer needed temporary buffers
    _domainParamsHaloZoneTmp.resize(0);
    _initialScatterDomainParams.resize(0);
    _initialScatterDomainMap.resize(0);

    double startTime = MPI_Wtime();

    // run the simulation
    for (size_t iter = 0; iter < mSimulationProps.getNumIterations(); iter++)
    {
        const bool current = iter & 1;
        const bool next = !current;

        // compute temperature halo zones (and the two most outer rows and columns of the tile)
        computeTempHaloZones(current, next);

        // start the halo zone exchange (async P2P communication)
        startHaloExchangeP2P();

        // compute the rest of the tile (inner part)
        computeTempTile(current, next);

        // wait for all halo zone exchanges to finalize
        awaitHaloExchangeP2P();

        if (shouldStoreData(iter))
        {
            /**********************************************************************************************************************/
            /*                          Store the data into the output file using parallel or sequential IO.                      */
            /**********************************************************************************************************************/
            if (mSimulationProps.useParallelIO())
            {
                storeDataIntoFileParallel(mFileHandle, iter, _tempTiles[next].data());
            }
            else
            {
                storeDataIntoFileSequential(mFileHandle, iter, _tempTiles[next].data());
            }
        }

        if (shouldPrintProgress(iter) && shouldComputeMiddleColumnAverageTemperature())
        {
            // compute and print the middle column average temperature using reduction of partial averages
            computeAndPrintMidColAverageParallel(iter);
        }
    }
    double elapsedTime = MPI_Wtime() - startTime;

    // retrieve the final temperature from all the nodes to a single matrix
    gatherComputedTempData(mSimulationProps.getNumIterations() & 1, outResult);

    // compute the final average temperature and print the final report
    computeAndPrintMidColAverageSequential(elapsedTime, outResult);
}

bool ParallelHeatSolver::shouldComputeMiddleColumnAverageTemperature() const
{
    /**********************************************************************************************************************/
    /*                Return true if rank should compute middle column average temperature.                               */
    /**********************************************************************************************************************/

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

void ParallelHeatSolver::storeDataIntoFileSequential(hid_t fileHandle,
                                                     size_t iteration,
                                                     const float *globalData)
{
    storeDataIntoFile(fileHandle, iteration, globalData);
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

void ParallelHeatSolver::storeDataIntoFileParallel(hid_t fileHandle, size_t iteration, const float *localData)
{
    if (fileHandle == H5I_INVALID_HID)
    {
        return;
    }

#ifdef H5_HAVE_PARALLEL
    array<hsize_t, 2> gridSize{static_cast<hsize_t>(mMaterialProps.getEdgeSize()), static_cast<hsize_t>(mMaterialProps.getEdgeSize())};
    array<hsize_t, 2> localGridSize{static_cast<hsize_t>(_edgeSizes.localHeight), static_cast<hsize_t>(_edgeSizes.localWidth)};
    array<hsize_t, 2> tileOffset{static_cast<hsize_t>(_edgeSizes.localHeight * (_worldRank / _decomposition.nx)),
                                 static_cast<hsize_t>(_edgeSizes.localWidth * (_worldRank % _decomposition.nx))};
    array<hsize_t, 2> blockCount{1, 1};

    // Create new HDF5 group in the output file
    string groupName = "Timestep_" + to_string(iteration / mSimulationProps.getWriteIntensity());

    Hdf5GroupHandle groupHandle(H5Gcreate(fileHandle, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    
    {
        /**********************************************************************************************************************/
        /*                                Compute the tile offsets and sizes.                                                 */
        /*               Note that the X and Y coordinates are swapped (but data not altered).                                */
        /**********************************************************************************************************************/

        // Create new dataspace and dataset using it.
        static constexpr string_view dataSetName{"Temperature"};

        Hdf5PropertyListHandle datasetPropListHandle(H5Pcreate(H5P_DATASET_CREATE));
        H5Pset_chunk(datasetPropListHandle, 2, localGridSize.data());
        /**********************************************************************************************************************/
        /*                            Create dataset property list to set up chunking.                                        */
        /*                Set up chunking for collective write operation in datasetPropListHandle variable.                   */
        /**********************************************************************************************************************/

        Hdf5DataspaceHandle dataSpaceHandle(H5Screate_simple(2, gridSize.data(), nullptr));
        Hdf5DatasetHandle dataSetHandle(H5Dcreate(groupHandle, dataSetName.data(),
                                                  H5T_NATIVE_FLOAT, dataSpaceHandle,
                                                  H5P_DEFAULT, datasetPropListHandle,
                                                  H5P_DEFAULT));

        Hdf5DataspaceHandle memSpaceHandle(H5Screate_simple(2, localGridSize.data(), nullptr));
        //H5Sselect_hyperslab(memSpaceHandle, H5S_SELECT_SET, tileOffset.data(), nullptr, blockCount.data(), localGridSize.data());
        H5Sselect_hyperslab(dataSpaceHandle, H5S_SELECT_SET, tileOffset.data(), nullptr, blockCount.data(), localGridSize.data());
        //H5Sselect_hyperslab(dataSpaceHandle, H5S_SELECT_SET, tileOffset.data(), nullptr, localGridSize.data(), nullptr);

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
