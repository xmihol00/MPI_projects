#include <mpi.h>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    constexpr int NUMBER_OF_ROWS = 2;
    if (world_size < NUMBER_OF_ROWS)
    {
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    int row_size = world_size / NUMBER_OF_ROWS;
    int color, key;
    if (row_size * NUMBER_OF_ROWS < world_size && world_rank >= row_size * NUMBER_OF_ROWS)
    {
        color = MPI_UNDEFINED;
        key = MPI_UNDEFINED;
    }
    else
    {
        color = world_rank / row_size; // grouping continuous values in the same color, e.g. 0 == 0/3 == 1/3 == 2/3
        key = world_rank % row_size;   // ordering values in a group e.g. 3%3 == 0, 3%4 == 1, 3%5 == 2
    }

    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &row_comm);

    if (row_comm != MPI_COMM_NULL)
    {
        int row_rank, row_comm_size;
        MPI_Comm_size(row_comm, &row_comm_size);
        MPI_Comm_rank(row_comm, &row_rank);
        cout << "row size: " << row_comm_size << ", row rank: " << row_rank << ", color: " << color << ", world size: " << world_size << ", world rank: " << world_rank << endl;

        MPI_Comm_free(&row_comm);
    }
    MPI_Finalize();
}
