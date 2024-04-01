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
    void initializeGridTopology();
    void initializeDataTypes();
    void initializeGrid();
    void performInitialScatter();
    void computeHaloZones();
    void startHaloZonesExchange();
    void computeTile();
    void awaitHaloZonesExchange();
};

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    LifeSimulation simulation(argc, argv);
    simulation.run();

    MPI_Finalize();
    return 0;
}
