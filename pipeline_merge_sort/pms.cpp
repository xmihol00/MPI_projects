#include <stdio.h>
#include <mpi.h>
#include <iostream>
#include <string>

using namespace std;

class PipelineMergeSort
{
    public:
        enum SortDirection
        {
            ASCENDING,
            DESCENDING
        };

        PipelineMergeSort(int rank, int size, MPI_Comm top_comm, MPI_Comm bot_comm);
        ~PipelineMergeSort();

        void sort(SortDirection direction = ASCENDING);

    private:
        const int RANK;
        const int SIZE;
        const MPI_Comm TOP_COMM;
        const MPI_Comm BOT_COMM;
        const uint32_t INPUT_SIZE;
        const uint32_t OUTPUT_SIZE;
        const uint32_t ITERATIONS;

        uint8_t *_input_buffer;
        uint8_t *_output_buffer;

        bool _ping_pong = true;
        MPI_Status _top_status = {0, 0, 0, 0, 0};
        MPI_Status _bot_status = {0, 0, 0, 0, 0};

        void check_top_status();
        void check_bot_status();

        void input_process();
        template <SortDirection direction>
        void merge_process();
        template <SortDirection direction>
        void output_process();
};

PipelineMergeSort::PipelineMergeSort(int rank, int size, MPI_Comm top_comm, MPI_Comm bot_comm) : 
    RANK{rank}, SIZE{size}, TOP_COMM{top_comm}, BOT_COMM{bot_comm}, INPUT_SIZE{1U << (rank - 1)}, OUTPUT_SIZE{1U << rank}, ITERATIONS{1U << (size - rank - 1)}
{
    if (rank > 0)
    {
        _input_buffer = new uint8_t[INPUT_SIZE];
    }
    else
    {
        _input_buffer = nullptr;
    }
    
    if (RANK != size - 1)
    {
        _output_buffer = new uint8_t[OUTPUT_SIZE];
    }
    else
    {
        _output_buffer = nullptr;
    }
}

PipelineMergeSort::~PipelineMergeSort()
{
    if (_input_buffer != nullptr)
    {
        delete[] _input_buffer;
    }

    if (_output_buffer != nullptr)
    {
        delete[] _output_buffer;
    }
}

void PipelineMergeSort::check_top_status()
{
    if (_top_status.MPI_ERROR != MPI_SUCCESS)
    {
        cerr << "Error: Process " << RANK << " failed to receive message from TOP input stream." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1); // error occurred during the data transfer
    }

    int size;
    MPI_Get_count(&_top_status, MPI_BYTE, &size);
    if (size != (int)INPUT_SIZE)
    {
        cerr << "Error: Process " << RANK << " received message of unexpected size from TOP input stream." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1); // error occurred during the data transfer
    }
}

void PipelineMergeSort::check_bot_status()
{
    if (_bot_status.MPI_ERROR != MPI_SUCCESS)
    {
        cerr << "Error: Process " << RANK << " failed to receive message from BOT input stream." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1); // error occurred during the data transfer
    }

    int size;
    MPI_Get_count(&_bot_status, MPI_BYTE, &size);
    if (size != 1)
    {
        cerr << "Error: Process " << RANK << " received message of unexpected size from BOT input stream." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1); // error occurred during the data transfer
    }
}

void PipelineMergeSort::sort(SortDirection direction)
{
    if (direction == ASCENDING)
    {
        if (RANK == 0)
        {
            input_process();
        }
        else if (RANK == SIZE - 1)
        {
            output_process<ASCENDING>();
        }
        else
        {
            merge_process<ASCENDING>();
        }
    }
    else
    {
        if (RANK == 0)
        {
            input_process();
        }
        else if (RANK == SIZE - 1)
        {
            output_process<DESCENDING>();
        }
        else
        {
            merge_process<DESCENDING>();
        }
    }
}

void PipelineMergeSort::input_process()
{
    union
    {
        uint8_t value;
        char character;
    } character_value;

    for (uint32_t i = 0; i < ITERATIONS; i++)
    {    
        // read the input value from stdin
        if (!cin.get(character_value.character)) // no more input values
        {
            cerr << "Error: Failed to read input value, possibly not enough input values present." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1); // sorting less values than expected is not implemented
        }
        
        if (_ping_pong)
        {
            MPI_Send(&character_value.value, 1, MPI_BYTE, 1, 0, TOP_COMM);
        }
        else
        {
            MPI_Send(&character_value.value, 1, MPI_BYTE, 1, 0, BOT_COMM);
        }

        _ping_pong = !_ping_pong;
    }
    
    cin.get(character_value.character);
    if (!cin.eof())
    {
        cerr << "Warning: There are still input values present, which will not be sorted." << endl;
    }
}

template <PipelineMergeSort::SortDirection direction>
void PipelineMergeSort::merge_process()
{
    uint8_t value;
    for (uint32_t n = 0; n < ITERATIONS; n++)
    {
        MPI_Recv(_input_buffer, INPUT_SIZE, MPI_BYTE, RANK - 1, 0, TOP_COMM, &_top_status);
        check_top_status();

        uint32_t i = 0, j = 0, k = 0;
        if (_ping_pong)
        {
            while (true)
            {
                if (j++ < INPUT_SIZE)
                {
                    MPI_Recv(&value, 1, MPI_BYTE, RANK - 1, 0, BOT_COMM, &_bot_status);
                    check_bot_status();
                }
                else
                {
                    break;
                }

                while (i < INPUT_SIZE && ((direction == ASCENDING && _input_buffer[i] < value) || (direction == DESCENDING && _input_buffer[i] > value)))
                {
                    _output_buffer[k++] = _input_buffer[i++];
                }
                _output_buffer[k++] = value;
            }

            while (i < INPUT_SIZE)
            {
                _output_buffer[k++] = _input_buffer[i++];
            }

            MPI_Send(_output_buffer, OUTPUT_SIZE, MPI_BYTE, RANK + 1, 0, TOP_COMM);
        }
        else
        {
            while (true)
            {
                if (j++ < INPUT_SIZE)
                {
                    MPI_Recv(&value, 1, MPI_BYTE, RANK - 1, 0, BOT_COMM, &_bot_status);
                    check_bot_status();
                }
                else
                {
                    break;
                }

                while (i < INPUT_SIZE && ((direction == ASCENDING && _input_buffer[i] < value) || (direction == DESCENDING && _input_buffer[i] > value)))
                {
                    MPI_Send(&_input_buffer[i++], 1, MPI_BYTE, RANK + 1, 0, BOT_COMM);
                }
                MPI_Send(&value, 1, MPI_BYTE, RANK + 1, 0, BOT_COMM);
            }

            while (i < INPUT_SIZE)
            {
                MPI_Send(&_input_buffer[i++], 1, MPI_BYTE, RANK + 1, 0, BOT_COMM);
            }
        }

        _ping_pong = !_ping_pong;
    }
}

template <PipelineMergeSort::SortDirection direction>
void PipelineMergeSort::output_process()
{
    uint8_t value;
    MPI_Recv(_input_buffer, INPUT_SIZE, MPI_BYTE, RANK - 1, 0, TOP_COMM, &_top_status);
    check_top_status();

    uint32_t i = 0, j = 0;
    while (true)
    {
        if (j++ < INPUT_SIZE)
        {
            MPI_Recv(&value, 1, MPI_BYTE, RANK - 1, 0, BOT_COMM, &_bot_status);
            check_bot_status();
        }
        else
        {
            break;
        }

        while (i < INPUT_SIZE && ((direction == ASCENDING && _input_buffer[i] < value) || (direction == DESCENDING && _input_buffer[i] > value)))
        {
            cout << (unsigned)_input_buffer[i++] << endl;
        }
        cout << (unsigned)value << endl;
    }

    while (i < INPUT_SIZE)
    {
        cout << (unsigned)_input_buffer[i++] << endl;
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv); // initialize the MPI library
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get the process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get the number of processes

    PipelineMergeSort::SortDirection direction = PipelineMergeSort::ASCENDING;
    if (argc > 1) // parse the command line arguments
    {
        string arg = argv[1];
        if (arg == "-d")
        {
            direction = PipelineMergeSort::DESCENDING;
        }
        
        if (rank == 0 && ((arg != "-a" && arg != "-d") || argc > 2))
        {
            cerr << "Warning: Invalid argument(s), expected '-a' for ASCENDING sort or '-d' for DESCENDING sort." << endl;
            cerr << "         ASCENDING sort will be performed by default." << endl;
        }
    }

    MPI_Comm top_comm, bot_comm;
    // simulate the top channel with a separate communicator, the top channel will be always used to send a whole batch of data, 
    // i.e. 2^rank elements, since the rank+1 process can only start computing after the rank process has finished 2^rank elements
    MPI_Comm_dup(MPI_COMM_WORLD, &top_comm);
    // simulate the bottom channel with another separate communicator, the bottom channel will be always used to send values one by one
    MPI_Comm_dup(MPI_COMM_WORLD, &bot_comm);

    PipelineMergeSort pms(rank, size, top_comm, bot_comm); // initialize the pipeline sorter
    pms.sort(direction);                                   // perform the sorting

    // clean up
    MPI_Comm_free(&top_comm);
    MPI_Comm_free(&bot_comm);
    MPI_Finalize();

    return 0;
}