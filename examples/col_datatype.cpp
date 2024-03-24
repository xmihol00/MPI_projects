#include <mpi.h>
#include <iostream>
#include <iomanip>

using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    constexpr int N = 5;
    constexpr int START_COL = 1;
    constexpr int NUM_COLS = 3;
    int matrix[N * N] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };
    int cols[N * N] = {0,};

    MPI_Datatype col_type;
    MPI_Type_vector(N, NUM_COLS, N, MPI_INT, &col_type);
    MPI_Type_commit(&col_type);

    if (world_rank == 0)
    {
        MPI_Send(matrix + START_COL, 1, col_type, 1, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 1)
    {
        MPI_Recv(cols + START_COL, 1, col_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                cout << setw(2) << cols[i * N + j] << " ";
            }
            cout << "\n";
        } 
    }

    MPI_Type_free(&col_type);
    MPI_Finalize();
}
