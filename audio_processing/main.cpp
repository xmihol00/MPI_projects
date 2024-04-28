#include "mian.h"

using namespace std;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    if (worldRank == 0)
    {
        cout << "Starting client" << endl;
        Client client(argc, argv);
        client.run();
    }
    else if (worldRank == 1)
    {
        cout << "Starting low pass server" << endl;
        LowPassServer server(argc, argv);
        server.run();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}