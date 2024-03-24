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

    constexpr int NUMBER_OF_COLS = 3;
    if (world_size < NUMBER_OF_COLS)
    {
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    int col_size = world_size / NUMBER_OF_COLS;
    int color, key;
    if (col_size * NUMBER_OF_COLS < world_size && world_rank >= col_size * NUMBER_OF_COLS)
    {
        color = MPI_UNDEFINED;
        key = MPI_UNDEFINED;
    }
    else
    {
        color = world_rank % NUMBER_OF_COLS; // grouping values with stride 'col_size', e.g. 0, 3, 6, ... for col_size == 3
        key = world_rank / col_size;   // defining the rank per rows, i.e. all processes in the first row are rank 0 in the new column, in second row rank 1 etc.
    }

    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &col_comm);

    if (col_comm != MPI_COMM_NULL)
    {
        int col_rank, col_comm_size;
        MPI_Comm_size(col_comm, &col_comm_size);
        MPI_Comm_rank(col_comm, &col_rank);
        cout << "col size: " << col_comm_size << ", col rank: " << col_rank << ", color: " << color << ", world size: " << world_size << ", world rank: " << world_rank << endl;

        MPI_Comm_free(&col_comm);
    }
    MPI_Finalize();
}
