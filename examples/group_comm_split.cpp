#include <mpi.h>
#include <iostream>
#include <bits/stdc++.h>
#include <vector>

using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Group old_group;
    MPI_Comm_group(MPI_COMM_WORLD, &old_group);
    MPI_Group new_group;
    vector<int> included_ranks;
    for (int i = 0; i < world_size; i++)
    {
        if (__builtin_popcount(i) == 1)
        {
            included_ranks.push_back(i);
        }
    }
    MPI_Group_incl(old_group, included_ranks.size(), included_ranks.data(), &new_group);
    MPI_Comm group_comm;
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &group_comm);

    if (group_comm != MPI_COMM_NULL)
    {
        int group_rank, group_size;
        MPI_Comm_size(group_comm, &group_size);
        MPI_Comm_rank(group_comm, &group_rank);
        cout << "group size: " << group_size << ", group rank: " << group_rank << ", world rank: " << world_rank << endl;

        MPI_Comm_free(&group_comm);
    }
    MPI_Group_free(&new_group);
    MPI_Finalize();
}
