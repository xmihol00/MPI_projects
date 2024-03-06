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
    _edgeSizes.localHorizontal = _edgeSizes.global / _decomposition.nx;
    _edgeSizes.localVertical = _edgeSizes.global / _decomposition.ny;

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

    _tempTiles[0].resize(_edgeSizes.localHorizontal * _edgeSizes.localVertical);
    _tempTiles[1].resize(_edgeSizes.localHorizontal * _edgeSizes.localVertical);
    _domainParamsTile.resize(_edgeSizes.localHorizontal * _edgeSizes.localVertical);
    _domainMapTile.resize(_edgeSizes.localHorizontal * _edgeSizes.localVertical);

    _tempHaloZones[0].resize(_edgeSizes.localHorizontal * 4 + _edgeSizes.localVertical * 4);
    _tempHaloZones[1].resize(_edgeSizes.localHorizontal * 4 + _edgeSizes.localVertical * 4);
    _tempHaloZones[2].resize(_edgeSizes.localHorizontal * 4 + _edgeSizes.localVertical * 4);
    _domainMapHaloZones[0].resize(_edgeSizes.localHorizontal * 4 + _edgeSizes.localVertical * 4);
    _domainMapHaloZones[1].resize(_edgeSizes.localHorizontal * 4 + _edgeSizes.localVertical * 4);
    _domainParamsHaloZones[0].resize(_edgeSizes.localHorizontal * 4 + _edgeSizes.localVertical * 4);
    _domainParamsHaloZones[1].resize(_edgeSizes.localHorizontal * 4 + _edgeSizes.localVertical * 4);

    if (_initialScatterColComm != MPI_COMM_NULL)
    {
        _initialScatterTemp.resize(_edgeSizes.global * _edgeSizes.localVertical);
        _initialScatterDomainParams.resize(_edgeSizes.global * _edgeSizes.localVertical);
        _initialScatterDomainMap.resize(_edgeSizes.global * _edgeSizes.localVertical);
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

void ParallelHeatSolver::computeHaloZones(const float *oldTemp, float *newTemp)
{
    /**********************************************************************************************************************/
    /*  Compute new temperatures in halo zones, so that copy operations can be overlapped with inner region computation.  */
    /*                        Use updateTile method to compute new temperatures in halo zones.                            */
    /*                             TAKE CARE NOT TO COMPUTE THE SAME AREAS TWICE                                          */
    /**********************************************************************************************************************/
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

void ParallelHeatSolver::run(std::vector<float, AlignedAllocator<float>> &outResult)
{
    std::array<MPI_Request, 8> requestsP2P{};
    
    const float *scatteredTempRow = mMaterialProps.getInitialTemperature().data();
    const float *scatteredDomainParamsRow = mMaterialProps.getDomainParameters().data();
    const int *scatteredDomainMapRow = mMaterialProps.getDomainMap().data();

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

    if (_initialScatterColComm != MPI_COMM_NULL)
    {
        MPI_Scatter(mMaterialProps.getInitialTemperature().data(), _edgeSizes.global * _edgeSizes.localVertical, MPI_FLOAT,
                    _initialScatterTemp.data(), _edgeSizes.global * _edgeSizes.localVertical, MPI_FLOAT, 0, _initialScatterColComm);
        scatteredTempRow = _initialScatterTemp.data();

        MPI_Scatter(mMaterialProps.getDomainParameters().data(), _edgeSizes.global * _edgeSizes.localVertical, MPI_FLOAT,
                    _initialScatterDomainParams.data(), _edgeSizes.global * _edgeSizes.localVertical, MPI_FLOAT, 0, _initialScatterColComm);
        scatteredDomainParamsRow = _initialScatterDomainParams.data();
        
        MPI_Scatter(mMaterialProps.getDomainMap().data(), _edgeSizes.global * _edgeSizes.localVertical, MPI_INT,
                    _initialScatterDomainMap.data(), _edgeSizes.global * _edgeSizes.localVertical, MPI_INT, 0, _initialScatterColComm);
        scatteredDomainMapRow = _initialScatterDomainMap.data();
    }

    for (size_t i = 0, j = 0; i < _edgeSizes.localVertical * _edgeSizes.global; i += _edgeSizes.global, j += _edgeSizes.localHorizontal)
    {
        MPI_Scatter(scatteredTempRow + i, _edgeSizes.localHorizontal, MPI_FLOAT,
                    _tempTiles[0].data() + j, _edgeSizes.localHorizontal, MPI_FLOAT, 0, _initialScatterRowComm);
        MPI_Scatter(scatteredDomainParamsRow + i, _edgeSizes.localHorizontal, MPI_FLOAT,
                    _domainParamsTile.data() + j, _edgeSizes.localHorizontal, MPI_FLOAT, 0, _initialScatterRowComm);
        MPI_Scatter(scatteredDomainMapRow + i, _edgeSizes.localHorizontal, MPI_INT,
                    _domainMapTile.data() + j, _edgeSizes.localHorizontal, MPI_INT, 0, _initialScatterRowComm);
    }
    
    // copy initial temperature to the second buffer
    copy(_tempTiles[0].begin(), _tempTiles[0].end(), _tempTiles[1].begin());

    // copy to halo zones North and South
    size_t haloHorizontalOffset = 2 * _edgeSizes.localHorizontal;
    copy(_tempTiles[0].begin(), _tempTiles[0].begin() + haloHorizontalOffset, _tempHaloZones[0].begin());
    copy(_tempTiles[0].end() - haloHorizontalOffset, _tempTiles[0].end(), _tempHaloZones[0].begin() + haloHorizontalOffset);

    copy(_domainMapHaloZones[0].begin(), _domainMapHaloZones[0].begin() + haloHorizontalOffset, _domainMapTile.begin());
    copy(_domainMapHaloZones[0].end() - haloHorizontalOffset, _domainMapHaloZones[0].end(), _domainMapTile.begin() + haloHorizontalOffset);

    copy(_domainParamsHaloZones[0].begin(), _domainParamsHaloZones[0].begin() + haloHorizontalOffset, _domainParamsTile.begin());
    copy(_domainParamsHaloZones[0].end() - haloHorizontalOffset, _domainParamsHaloZones[0].end(), _domainParamsTile.begin() + haloHorizontalOffset);

    // copy to halo zones West and East
    size_t haloVerticalOffset = 2 * _edgeSizes.localVertical;
    for (size_t i = 0; i < _edgeSizes.localVertical; i++)
    {
        _tempHaloZones[0][2 * haloHorizontalOffset + 2 * i] = _tempTiles[0][i * _edgeSizes.localHorizontal];
        _tempHaloZones[0][2 * haloHorizontalOffset + 2 * i + 1] = _tempTiles[0][i * _edgeSizes.localHorizontal + 1];
        _tempHaloZones[0][2 * haloHorizontalOffset + haloVerticalOffset + 2 * i] = _tempTiles[0][i * _edgeSizes.localHorizontal + _edgeSizes.localHorizontal - 2];
        _tempHaloZones[0][2 * haloHorizontalOffset + haloVerticalOffset + 2 * i + 1] = _tempTiles[0][i * _edgeSizes.localHorizontal + _edgeSizes.localHorizontal - 1];

        _domainMapHaloZones[0][2 * haloHorizontalOffset + 2 * i] = _domainMapTile[i * _edgeSizes.localHorizontal];
        _domainMapHaloZones[0][2 * haloHorizontalOffset + 2 * i + 1] = _domainMapTile[i * _edgeSizes.localHorizontal + 1];
        _domainMapHaloZones[0][2 * haloHorizontalOffset + haloVerticalOffset + 2 * i] = _domainMapTile[i * _edgeSizes.localHorizontal + _edgeSizes.localHorizontal - 2];
        _domainMapHaloZones[0][2 * haloHorizontalOffset + haloVerticalOffset + 2 * i + 1] = _domainMapTile[i * _edgeSizes.localHorizontal + _edgeSizes.localHorizontal - 1];

        _domainParamsHaloZones[0][2 * haloHorizontalOffset + 2 * i] = _domainParamsTile[i * _edgeSizes.localHorizontal];
        _domainParamsHaloZones[0][2 * haloHorizontalOffset + 2 * i + 1] = _domainParamsTile[i * _edgeSizes.localHorizontal + 1];
        _domainParamsHaloZones[0][2 * haloHorizontalOffset + haloVerticalOffset + 2 * i] = _domainParamsTile[i * _edgeSizes.localHorizontal + _edgeSizes.localHorizontal - 2];
        _domainParamsHaloZones[0][2 * haloHorizontalOffset + haloVerticalOffset + 2 * i + 1] = _domainParamsTile[i * _edgeSizes.localHorizontal + _edgeSizes.localHorizontal - 1];
    }

    // parameters for all to all gather
    int sendCounts[4] = {haloHorizontalOffset, haloHorizontalOffset, haloVerticalOffset, haloVerticalOffset};
    int displacements[4] = {0, haloHorizontalOffset, 2 * haloHorizontalOffset, 2 * haloHorizontalOffset + haloVerticalOffset};

    MPI_Barrier(MPI_COMM_WORLD);

    printHalo(0, 0);
    printHalo(1, 0);
    printHalo(2, 0);
    printHalo(3, 0);
    
    // gather the halo zones across the ranks
    MPI_Neighbor_alltoallv(_tempHaloZones[0].data(), sendCounts, displacements, MPI_FLOAT,
                           _tempHaloZones[1].data(), sendCounts, displacements, MPI_FLOAT, _topologyComm);
    MPI_Neighbor_alltoallv(_domainMapHaloZones[0].data(), sendCounts, displacements, MPI_INT,
                           _domainMapHaloZones[1].data(), sendCounts, displacements, MPI_INT, _topologyComm);
    MPI_Neighbor_alltoallv(_domainParamsHaloZones[0].data(), sendCounts, displacements, MPI_FLOAT,
                           _domainParamsHaloZones[1].data(), sendCounts, displacements, MPI_FLOAT, _topologyComm);

    printTile(0);
    printTile(1);
    printTile(2);
    printTile(3);

    printHalo(0, 1);
    printHalo(1, 1);
    printHalo(2, 1);
    printHalo(3, 1);

    /**********************************************************************************************************************/
    /*                                         Scatter initial data.                                                      */
    /**********************************************************************************************************************/

    /**********************************************************************************************************************/
    /* Exchange halo zones of initial domain temperature and parameters using P2P communication. Wait for them to finish. */
    /**********************************************************************************************************************/

    /**********************************************************************************************************************/
    /*                            Copy initial temperature to the second buffer.                                          */
    /**********************************************************************************************************************/

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
