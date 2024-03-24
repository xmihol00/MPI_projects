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
    
    struct Values
    {
        double d1;
        char   c1;
        int    i1;
        float  f1;
    };

    Values structs[ALLOWED_WORLD_SIZE] = {
        {.d1 = 1.0f, .c1 = 2, .i1 = 3, .f1 = 4.0f},
        {.d1 = 5.0f, .c1 = 6, .i1 = 7, .f1 = 8.0f},
        {.d1 = 9.0f, .c1 = 10, .i1 = 11, .f1 = 12.0f},
        {.d1 = 13.0f, .c1 = 14, .i1 = 15, .f1 = 16.0f}
    };

    Values local = {.d1 = 0, .c1 = 0, .i1 = 0, .f1 = 0};

    MPI_Aint add_d1;
    MPI_Get_address(&(structs[0].d1), &add_d1); // !!!
    MPI_Aint add_c1;
    MPI_Get_address(&(structs[0].c1), &add_c1); // !!!
    MPI_Aint add_i1;
    MPI_Get_address(&(structs[0].i1), &add_i1); // !!!
    MPI_Aint add_f1;
    MPI_Get_address(&(structs[0].f1), &add_f1); // !!!

    MPI_Datatype types[2] = {MPI_CHAR, MPI_INT};
    int lengths[2] = {1, 1};
    MPI_Aint offsets[2] = {add_c1 - add_d1, add_i1 - add_d1};

    MPI_Datatype struct_type;
    MPI_Type_create_struct(2, lengths, offsets, types, &struct_type);
    MPI_Type_commit(&struct_type);

    MPI_Datatype struct_resized_type;
    MPI_Type_create_resized(struct_type, 0, sizeof(Values), &struct_resized_type); // !!!
    MPI_Type_commit(&struct_resized_type);

    MPI_Scatter(structs, 1, struct_resized_type, &local, 1, struct_type, 0, MPI_COMM_WORLD);
    for (int i = 0; i < ALLOWED_WORLD_SIZE; i++)
    {
        if (world_rank == i)
        {
            cout << "rank " << world_rank << ": " << local.d1 << " " << (int)local.c1 << " " << local.i1 << " " << local.f1 << "\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Type_free(&struct_type);
    MPI_Type_free(&struct_resized_type);
    MPI_Finalize();
}
