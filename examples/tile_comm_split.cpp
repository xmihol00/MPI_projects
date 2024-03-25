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

    constexpr int TILE_HEIGHT = 3;
    constexpr int TILE_WIDTH = 3;
    constexpr int TILES_PER_ROW = 2;
    if (world_size < TILE_HEIGHT * TILE_WIDTH * TILES_PER_ROW)
    {
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    int max_tiles = world_size / (TILE_HEIGHT * TILE_WIDTH * TILES_PER_ROW);
    int color, key;
    if (max_tiles * TILE_HEIGHT * TILE_WIDTH * TILES_PER_ROW < world_size && world_rank >= max_tiles * TILE_HEIGHT * TILE_WIDTH * TILES_PER_ROW)
    {
        color = MPI_UNDEFINED;
        key = MPI_UNDEFINED;
    }
    else
    {
        int tile_row = world_rank / (TILE_HEIGHT * TILE_WIDTH * TILES_PER_ROW);
        int tile_col = (world_rank % (TILE_WIDTH * TILES_PER_ROW)) / TILE_WIDTH;
        int row = world_rank / (TILE_WIDTH * TILES_PER_ROW);
        int col = world_rank % TILE_WIDTH;
        color = tile_row * TILES_PER_ROW + tile_col;
        key = row * TILE_WIDTH + col;
    }

    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &row_comm);

    if (row_comm != MPI_COMM_NULL)
    {
        int tile_rank, tile_comm_size;
        MPI_Comm_size(row_comm, &tile_comm_size);
        MPI_Comm_rank(row_comm, &tile_rank);
        cout << "tile size: " << tile_comm_size << ", tile rank: " << tile_rank << ", color: " << color << ", world size: " << world_size << ", world rank: " << world_rank << endl;

        MPI_Comm_free(&row_comm);
    }
    MPI_Finalize();
}
