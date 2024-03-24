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
    constexpr int START_ROW = 1;
    constexpr int NUM_ROWS = 3;
    int matrix[N * N] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };
    int rows[N * N] = {0,};

    MPI_Datatype row_type;
    MPI_Type_contiguous(N, MPI_INT, &row_type);
    MPI_Type_commit(&row_type);

    if (world_rank == 0)
    {
        MPI_Send(matrix + START_ROW * N, NUM_ROWS, row_type, 1, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 1)
    {
        MPI_Recv(rows + START_ROW * N, NUM_ROWS, row_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                cout << setw(2) << rows[i * N + j] << " ";
            }
            cout << "\n";
        } 
    }

    MPI_Type_free(&row_type);
    MPI_Finalize();
}
