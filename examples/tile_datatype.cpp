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
    constexpr int NUM_DIMS = 2;
    constexpr int TOP_OFFSET = 2;
    constexpr int LEFT_OFFSET = 1;
    constexpr int HEIGHT = 3;
    constexpr int WIDTH = 3;
    constexpr int SIZES[NUM_DIMS] = {N, N};
    constexpr int SUB_SIZES[NUM_DIMS] = {HEIGHT, WIDTH};
    constexpr int STARTS[NUM_DIMS] = {TOP_OFFSET, LEFT_OFFSET};

    int matrix[N * N] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };
    int tile[N * N] = {0,};

    MPI_Datatype tile_type;
    MPI_Type_create_subarray(NUM_DIMS, SIZES, SUB_SIZES, STARTS, MPI_ORDER_C, MPI_INT, &tile_type);
    MPI_Type_commit(&tile_type);

    if (world_rank == 0)
    {
        MPI_Send(matrix, 1, tile_type, 1, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 1)
    {
        MPI_Recv(tile, 1, tile_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                cout << setw(2) << tile[i * N + j] << " ";
            }
            cout << "\n";
        } 
    }

    MPI_Type_free(&tile_type);
    MPI_Finalize();
}
