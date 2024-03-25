#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <time.h>

using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    srand(world_rank * time(0));
    int num_elements = rand() % (world_size * 2);
    for (int i = 0; i < world_size; i++)
    {
        if (i == world_rank)
        {
            cout << "rank: " << world_rank << ", elements: " << num_elements << endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    vector<int> nums;
    nums.resize(num_elements);
    fill(nums.begin(), nums.end(), world_rank);

    int reduce = 0;
    MPI_Reduce(&num_elements, &reduce, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    vector<int> packed;
    if (world_rank == 0)
    {
        packed.resize(reduce);
    }

    int index = 0;
    MPI_Exscan(&num_elements, &index, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    vector<int> counts;
    vector<int> displacements;
    if (world_rank == 0)
    {
        counts.resize(world_size);
        displacements.resize(world_size);
    }
    MPI_Gather(&num_elements, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&index, 1, MPI_INT, displacements.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(nums.data(), num_elements, MPI_INT, packed.data(), counts.data(), displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        for (const int &value : packed)
        {
            cout << value << " ";
        }
        cout << endl;
    }
    MPI_Finalize();
}
