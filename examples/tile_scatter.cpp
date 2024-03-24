#include <mpi.h>
#include <iostream>
#include <iomanip>

using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    constexpr int ALLOWED_WORLD_SIZE = 4;
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_size != ALLOWED_WORLD_SIZE)
    {
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    constexpr int N = 4;
    constexpr int M = 6;
    constexpr int NUM_DIMS = 2;
    constexpr int TOP_OFFSET = 0;
    constexpr int LEFT_OFFSET = 0;
    constexpr int HEIGHT = 2;
    constexpr int WIDTH = 3;
    constexpr int SIZES[NUM_DIMS] = {N, M};
    constexpr int SUB_SIZES[NUM_DIMS] = {HEIGHT, WIDTH};
    constexpr int STARTS[NUM_DIMS] = {TOP_OFFSET, LEFT_OFFSET};

    int matrix[N * M] = {
        1, 2, 3, 4, 5, 6, 
        7, 8, 9, 10, 11, 12, 
        13, 14, 15, 16, 17, 18, 
        19, 20, 21, 22, 23, 24
    };
    int tile[HEIGHT * WIDTH] = {0,};

    MPI_Datatype tile_type;
    MPI_Type_create_subarray(NUM_DIMS, SIZES, SUB_SIZES, STARTS, MPI_ORDER_C, MPI_INT, &tile_type);
    MPI_Type_commit(&tile_type);
    MPI_Datatype tile_resized_type;
    MPI_Type_create_resized(tile_type, 0, WIDTH * sizeof(int), &tile_resized_type);
    MPI_Type_commit(&tile_resized_type);

    int counts[ALLOWED_WORLD_SIZE];
    int displacements[ALLOWED_WORLD_SIZE];
    for (int i = 0; i < ALLOWED_WORLD_SIZE; i++)
    {
        counts[i] = 1;
        int row = i / 2;
        int col = i % 2;
        displacements[i] = row * HEIGHT * (M / WIDTH) + col; // !!!
    }

    MPI_Scatterv(matrix, counts, displacements, tile_resized_type, tile, HEIGHT * WIDTH, MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < ALLOWED_WORLD_SIZE; i++)
    {
        if (world_rank == i)
        {
            cout << "rank " << world_rank << ":\n";
            for (int j = 0; j < HEIGHT; j++)
            {
                for (int k = 0; k < WIDTH; k++)
                {
                    cout << setw(2) << tile[j * WIDTH + k] << " ";
                }
                cout << "\n";
            }
            cout << "\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Type_free(&tile_type);
    MPI_Type_free(&tile_resized_type);
    MPI_Finalize();
}
