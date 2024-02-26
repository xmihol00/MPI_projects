// =======================================================================================================================================================
// Algorithm:   Pipeline Merge Sort
// Author:      David Mihola
// E-mail:      xmihol00@stud.fit.vutbr.cz
// Date:        25. 2. 2024
// Description: An implementation of Pipeline Merge Sort using the MPI library for communication. 
// =======================================================================================================================================================

// =======================================================================================================================================================
// compile with: mpic++ -std=c++17 pms.cpp -o pms
// run with:     mpirun -np <number of processes> ./pms [-a|-d]
// test with:
//               D=""      # choose a sort direction, ASCENDING is by default
//               N=8       # choose a number of processes larger than 1
//               M=$((2**($N-1)))
//               if [ "$D" = "-d" ]; then R="-r"; else R=""; fi
//               dd if=/dev/random bs=1 count=$M 2>/dev/null | mpirun --oversubscribe -np $N ./pms $D | sort -nc $R && echo "SORTED" || echo "NOT SORTED"
// =======================================================================================================================================================

#include <stdio.h>
#include <mpi.h>
#include <iostream>
#include <string>

using namespace std;

enum SortDirection
{
    ASCENDING,
    DESCENDING
};

enum CommunicationStyles
{
    QUEUE,
    BATCH_SYNC,
    BATCH_ASYNC
};

/**
 * @brief Sorts a sequence of 2^(N-1) values using the pipeline merge sort algorithm with N processes.
 *        The input values are read from STDIN and the sorted sequence is written to STDOUT.
 *        Only sequences of a length equal to 2^(N-1) are allowed for N processes.
 */
template <CommunicationStyles communication_style = CommunicationStyles::BATCH_SYNC>
class PipelineMergeSort
{
    public:
        /**
         * @brief Construct a new Pipeline Merge Sort object.
         * @param rank The process ID, must be obtained by a call to MPI_Comm_rank.
         *             Must be in the range [0, size-1].
         * @param size The number of processes, must be obtained by a call to MPI_Comm_size.
         *             Must be in the range [1, INT_MAX].
         * @param top_comm The top channel/stream communicator, must be obtained by a call to MPI_Comm_dup of the MPI_COMM_WORLD communicator. 
         *                 Cannot be the same as bot_comm.
         * @param bot_comm The bottom channel/stream communicator, must be obtained by a call to MPI_Comm_dup of the MPI_COMM_WORLD communicator.
         *                 Cannot be the same as top_comm.
         */
        PipelineMergeSort(int rank, int size, MPI_Comm top_comm, MPI_Comm bot_comm);

        /**
         * @brief Destroy the Pipeline Merge Sort object.
         */
        ~PipelineMergeSort();
        
        /**
         * @brief Perform the sorting of input values read from STDIN and write the sorted sequence to STDOUT.
         * @param direction The direction of the sorting, either ASCENDING or DESCENDING.
         */
        void sort(SortDirection direction = ASCENDING);

    private:
        const int RANK;                 // process ID
        const int SIZE;                 // number of processes
        const MPI_Comm TOP_COMM;        // top channel/stream communicator
        const MPI_Comm BOT_COMM;        // bottom channel/stream communicator
        const uint32_t INPUT_SIZE;      // size of the input buffer and the input which must be received from the previous process before sorting can start
        const uint32_t OUTPUT_SIZE;     // size of the output buffer and the output which, will be send as a batch every second iteration
        const uint32_t ITERATIONS;      // number of iterations to perform the sorting, decreases by half with increasing rank

        uint8_t *_input_buffer;         // input buffer to receive values from the previous process
        uint8_t *_output_buffer;        // output buffer to send values to the next process

        bool _ping_pong = true;         // flag to indicate which channel/stream to use as output

        // status objects to check the result of the MPI communication
        MPI_Status _top_status = {0, 0, 0, 0, 0};
        MPI_Status _bot_status = {0, 0, 0, 0, 0};

        // helper functions to check the status of the MPI communication
        void check_top_status();
        void check_bot_status();

        void input_process(); // first process (rank == 0) reads the input values from STDIN and sends them in an alternating fashion to the 2nd process

        // let the compiler optimize the code for the specified sort direction
        template <SortDirection direction>
        void merge_process(); // processes with ranks between 1 and size-2 receive values from the previous process in batches of size 2^(rank-1), 
                              // merge sort them into batches of size 2^rank and send them to the next process
        template <SortDirection direction>
        void output_process(); // last process (rank == size-1) receives values from the previous process, merge sorts them into a single sequence and 
                               // writes them to STDOUT
};

template<CommunicationStyles communication_style>
PipelineMergeSort<communication_style>::PipelineMergeSort(int rank, int size, MPI_Comm top_comm, MPI_Comm bot_comm) : 
    RANK{rank}, SIZE{size}, TOP_COMM{top_comm}, BOT_COMM{bot_comm}, 
    INPUT_SIZE{1U << (rank - 1)},        // 2^(rank-1)
    OUTPUT_SIZE{1U << rank},             // 2^rank
    ITERATIONS{1U << (size - rank - 1)}  // 2^(size-rank-1)
{
    if (size < 0 || rank < 0 || rank >= size || top_comm == bot_comm) // invalid constructor arguments
    {
        cerr << "Error: Invalid arguments for the PipelineMergeSort constructor." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1); // invalid arguments
    }

    if (rank > 0) // all processes except the first process
    {
        _input_buffer = new uint8_t[INPUT_SIZE];
    }
    else // first process does not need an input buffer
    {
        _input_buffer = nullptr;
    }
    
    if (RANK != size - 1) // all processes except the last process
    {
        _output_buffer = new uint8_t[OUTPUT_SIZE];
    }
    else // last process does not need an output buffer
    {
        _output_buffer = nullptr;
    }
}

template<CommunicationStyles communication_style>
PipelineMergeSort<communication_style>::~PipelineMergeSort()
{
    // clean up allocated memory

    if (_input_buffer != nullptr) // all processes except the first process
    {
        delete[] _input_buffer;
    }

    if (_output_buffer != nullptr) // all processes except the last process
    {
        delete[] _output_buffer;
    }
}

template<CommunicationStyles communication_style>
void PipelineMergeSort<communication_style>::check_top_status()
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
        MPI_Abort(MPI_COMM_WORLD, 1); // not enough values received
    }
}

template<CommunicationStyles communication_style>
void PipelineMergeSort<communication_style>::check_bot_status()
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
        MPI_Abort(MPI_COMM_WORLD, 1); // not enough values received
    }
}

template<CommunicationStyles communication_style>
void PipelineMergeSort<communication_style>::sort(SortDirection direction)
{
    // decide the direction of the sorting in advance to be able to use compile-time templates for optimization
    if (direction == ASCENDING)
    {
        if (RANK == 0) // first process
        {
            input_process();
        }
        else if (RANK == SIZE - 1) // last process
        {
            output_process<ASCENDING>();
        }
        else // all other processes
        {
            merge_process<ASCENDING>();
        }
    }
    else
    {
        if (RANK == 0) // first process
        {
            input_process();
        }
        else if (RANK == SIZE - 1) // last process
        {
            output_process<DESCENDING>();
        }
        else // all other processes
        {
            merge_process<DESCENDING>();
        }
    }
}

template<CommunicationStyles communication_style>
void PipelineMergeSort<communication_style>::input_process()
{
    union
    {
        uint8_t value;
        char character;
    } character_value; // dirty trick to cast values between char and uint8_t

    for (uint32_t i = 0; i < ITERATIONS; i++)
    {    
        // read the input value from stdin
        if (!cin.get(character_value.character)) // no more input values
        {
            cerr << "Error: Failed to read input value, possibly not enough input values present." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1); // sorting less values than expected is not implemented
        }
        
        if (_ping_pong) // write to top channel/stream
        {
            MPI_Send(&character_value.value, 1, MPI_BYTE, 1, 0, TOP_COMM);
        }
        else // write to bottom channel/stream
        {
            MPI_Send(&character_value.value, 1, MPI_BYTE, 1, 0, BOT_COMM);
        }

        _ping_pong = !_ping_pong; // switch the channel/stream
    }
    
    cin.get(character_value.character); // try to read one more value
    if (!cin.eof()) // there are still input values present
    {
        cerr << "Warning: There are still input values present, which will not be sorted." << endl;
    }
}

template<CommunicationStyles communication_style> 
template<SortDirection direction>
void PipelineMergeSort<communication_style>::merge_process()
{
    uint8_t value;
    for (uint32_t n = 0; n < ITERATIONS; n++)
    {
        MPI_Recv(_input_buffer, INPUT_SIZE, MPI_BYTE, RANK - 1, 0, TOP_COMM, &_top_status); // wait for 2^(rank-1) values
        check_top_status();

        uint32_t i = 0, j = 0, k = 0; // i - top channel/stream index, j - bottom channel/stream index, k - output buffer index
        if (_ping_pong) // write to top channel/stream in batches
        {
            while (j++ < INPUT_SIZE) // read all values from the bottom channel/stream
            {
                MPI_Recv(&value, 1, MPI_BYTE, RANK - 1, 0, BOT_COMM, &_bot_status);
                check_bot_status();

                // read/write from the top channel/stream until the value from the bottom channel/stream is smaller (ASCENDING sort) or greater (DESCENDING sort)
                while (i < INPUT_SIZE && ((direction == ASCENDING && _input_buffer[i] < value) || (direction == DESCENDING && _input_buffer[i] > value)))
                {
                    _output_buffer[k++] = _input_buffer[i++];
                }

                // write the value from the bottom channel/stream, now it is smaller (ASCENDING sort) or greater (DESCENDING sort) 
                // than the value from the top channel/stream
                _output_buffer[k++] = value; 
            }

            while (i < INPUT_SIZE) // read/write the remaining values from the top channel/stream, if there are any left
            {
                _output_buffer[k++] = _input_buffer[i++];
            }

            // avoid sending values one after another, it is not efficient to send them at once as there is a lot more overhead caused by the MPI communication
            // for smaller batches (processes with lower ranks) the communication will be buffered, for larger batches (processes with higher ranks) the send
            // will be blocking, but the communication will be more efficient, as the initiation latency of the communication will be small in comparison to the
            // time needed to send the whole batch
            MPI_Send(_output_buffer, OUTPUT_SIZE, MPI_BYTE, RANK + 1, 0, TOP_COMM); // send the merge-sorted batch to the next process
        }
        else // write to bottom channel/stream, here it is not necessary to write values one after another, so the next process can start sorting
        {
            while (j++ < INPUT_SIZE) // read all values from the bottom channel/stream
            {
                MPI_Recv(&value, 1, MPI_BYTE, RANK - 1, 0, BOT_COMM, &_bot_status);
                check_bot_status();

                // send values from the top channel/stream until the value from the bottom channel/stream is smaller (ASCENDING sort) or greater (DESCENDING sort)
                while (i < INPUT_SIZE && ((direction == ASCENDING && _input_buffer[i] < value) || (direction == DESCENDING && _input_buffer[i] > value)))
                {
                    MPI_Send(&_input_buffer[i++], 1, MPI_BYTE, RANK + 1, 0, BOT_COMM);
                }

                // send the value from the bottom channel/stream, now it is smaller (ASCENDING sort) or greater (DESCENDING sort) 
                // than the value from the top channel/stream
                // this communication will be always buffered, meaning that the function returns before the receiving process has received the value,
                // which means that computation can be partly overlapped with the communication
                MPI_Send(&value, 1, MPI_BYTE, RANK + 1, 0, BOT_COMM);
            }

            while (i < INPUT_SIZE) // send the remaining values from the top channel/stream, if there are any left to be sent
            {
                MPI_Send(&_input_buffer[i++], 1, MPI_BYTE, RANK + 1, 0, BOT_COMM);
            }
        }

        _ping_pong = !_ping_pong; // switch the channel/stream
    }
}

template<CommunicationStyles communication_style> 
template<SortDirection direction>
void PipelineMergeSort<communication_style>::output_process()
{
    uint8_t value;
    MPI_Recv(_input_buffer, INPUT_SIZE, MPI_BYTE, RANK - 1, 0, TOP_COMM, &_top_status); // wait for half of the values to be received
    check_top_status();

    uint32_t i = 0, j = 0;
    while (j++ < INPUT_SIZE) // read all values from the bottom channel/stream
    {
        MPI_Recv(&value, 1, MPI_BYTE, RANK - 1, 0, BOT_COMM, &_bot_status);
        check_bot_status();

        // read/print from the top channel/stream until the value from the bottom channel/stream is smaller (ASCENDING sort) or greater (DESCENDING sort)
        while (i < INPUT_SIZE && ((direction == ASCENDING && _input_buffer[i] < value) || (direction == DESCENDING && _input_buffer[i] > value)))
        {
            cout << (unsigned)_input_buffer[i++] << endl;
        }

        // print the value from the bottom channel/stream, now it is smaller (ASCENDING sort) or greater (DESCENDING sort)
        // than the value from the top channel/stream
        cout << (unsigned)value << endl;
    }

    while (i < INPUT_SIZE) // print the remaining values from the top channel/stream, if there are any left to be printed
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

    if (size < 2) // at least two processes are needed
    {
        cerr << "Error: At least two processes are needed to perform the sorting." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1); // not enough processes
    }

    SortDirection direction = SortDirection::ASCENDING;
    if (argc > 1) // parse the command line arguments (optional)
    {
        string arg = argv[1];
        if (arg == "-d")
        {
            direction = SortDirection::DESCENDING;
        }
        
        if (rank == 0 && ((arg != "-a" && arg != "-d") || argc > 2)) // valid arguments are '-a' for ASCENDING sort and '-d' for DESCENDING sort
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

    PipelineMergeSort<> pms(rank, size, top_comm, bot_comm); // initialize the pipeline sorter
    pms.sort(direction);                                   // perform the sorting

    // clean up
    MPI_Comm_free(&top_comm);
    MPI_Comm_free(&bot_comm);
    MPI_Finalize();

    return 0;
}
