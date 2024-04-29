#include "main.h"

using namespace std;

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
        LowPassServer server(argc, argv);
        server.run();
    }

    MPI_Finalize();
    return 0;
}