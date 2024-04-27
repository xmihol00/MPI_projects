#include "mian.h"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    if (worldRank == 0)
    {
        Client client(argc, argv);
        client.run();
    }
    else if (worldRank == 1)
    {
        Server server(argc, argv);
        server.run();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}