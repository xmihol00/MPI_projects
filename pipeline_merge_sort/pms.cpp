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
// test with:    https://github.com/xmihol00/MPI_projects/blob/main/pipeline_merge_sort/test.sh or manually with the following commands:
//               M=123456  # choose a number of input values
//               D=""      # choose a sort direction, ASCENDING is by default
//               C=""      # choose a communication style, BATCH is by default
//               N=$(python3 -c "from math import ceil, log2; print(ceil(log2($M)+1), end='')")
//               if [ "$D" = "" ]; then D="-a"; else D="-d"; fi
//               if [ "$D" = "-d" ]; then R="-r"; else R=""; fi
//               dd if=/dev/random bs=1 count=$M 2>/dev/null | mpirun --oversubscribe -np $N ./pms $D | sort -nc $R && echo -e "\e[32mSORTED\e[0m" || echo -e "\e[31mNOT SORTED\e[0m"
// =======================================================================================================================================================

// =======================================================================================================================================================
// Theoretically the pipeline merge sort algorithm should run in O(M), where M is the number of input values. However, experimentally measured times do
// not support it. For M larger than 2^17 the run time starts to increase exponentially even on a machine with enough CPUs (cores). This is caused by the 
// amount of data required to be sent and received by the processes, as well as that the buffers will not fit into caches of processes with larger ranks.
// See the measurements at:
//      https://github.com/xmihol00/MPI_projects/tree/main/pipeline_merge_sort/performance_bara - for the Barbora supercomputer in Ostrava 36 cores per node
//      https://github.com/xmihol00/MPI_projects/tree/main/pipeline_merge_sort/performance_ntb  - for an 8 core laptop
//      https://github.com/xmihol00/MPI_projects/tree/main/pipeline_merge_sort/performance_pc   - for a 4 core desktop
// =======================================================================================================================================================

#include <stdio.h>
#include <mpi.h>
#include <iostream>
#include <string>
#include <queue>

using namespace std;

#if 0
    #define INFO_PRINT(rank, message) if (rank == 0) { cerr << "Info: " << message << endl; }
#else
    #define INFO_PRINT(rank, message) 
#endif

/**
 * @brief Possible sort directions.
 */
enum SortDirection
{
    ASCENDING,
    DESCENDING
};

/**
 * @brief Communication styles between processes. See the code below for more information.
 *        SINGLE - Values are sent one by one across both channels.
 *        BATCH  - Values are sent in batches of size 2^rank across the TOP channel to reduce number of messages as the rank+1 process can only start
 *                 merge-sorting, when it receives 2^rank values on the TOP channel. Values are sent one by one across the BOTTOM channel.
 */
enum CommunicationStyles
{
    SINGLE,
    BATCH,
};

/**
 * @brief Sorts a sequence of [(2^(N-2)+1), ..., (2^(N-1))] values using the pipeline merge sort algorithm with N processes.
 *        The input values are read from STDIN and the sorted sequence is written to STDOUT.
 */
template <CommunicationStyles communication_style = CommunicationStyles::BATCH>
class PipelineMergeSort
{
    public:
        /**
         * @brief Construct a new Pipeline Merge Sort object.
         * @param rank The process ID, must be obtained by a call to MPI_Comm_rank.
         *             Must be in the range [0, size-1].
         * @param size The number of processes, must be obtained by a call to MPI_Comm_size.
         *             Must be in the range [1, INT_MAX].
         * @param top_comm The top channel communicator, must be obtained by a call to MPI_Comm_dup of the MPI_COMM_WORLD communicator. 
         *                 Cannot be the same as bot_comm.
         * @param bot_comm The bottom stream communicator, must be obtained by a call to MPI_Comm_dup of the MPI_COMM_WORLD communicator.
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
        const MPI_Comm TOP_COMM;        // top channel communicator
        const MPI_Comm BOT_COMM;        // bottom stream communicator
        const uint64_t INPUT_SIZE;      // size of the input buffer and the input which must be received from the previous process before sorting can start
        const uint64_t OUTPUT_SIZE;     // size of the output buffer and the output which, will be send as a batch every second iteration
        const uint64_t ITERATIONS;      // number of iterations to perform the sorting, decreases by half with increasing rank

        // use simple arrays for the input and output buffers for the top channel, as it is necessary to send a whole batch before the next process can start
        uint8_t *_input_buffer;         // input buffer to receive values from the previous process
        uint8_t *_output_buffer;        // output buffer to send values to the next process

        // use a queue for the input values from the bottom channel, as it is necessary to read the values from the bottom channel as fast as possible
        queue<uint8_t> _input_queue;    // input queue to receive values from the previous process bottom stream

        bool _ping_pong = true;         // flag to indicate which channel/stream to use as output

        uint64_t _read_idx = 0;         // index of the yet not read value from the input buffer
        uint64_t _write_idx = 0;        // index of the yet not written value to the output buffer
        MPI_Request _top_send_request;  // request object used for non-blocking communication on the top channel

        // status objects to check the result of the MPI communication
        MPI_Status _top_status = {0, 0, 0, 0, 0};
        MPI_Status _bot_status = {0, 0, 0, 0, 0};

        // helper functions to check the status of the MPI communication and terminate the program if an error occurs
        void check_top_status(int size);
        void check_bot_status();

        void setup_top_channel();       // receive 2^(rank-1) values from the previous process in a batch (BATCH communication style),  
                                        // or fill the batch one by one (SINGLE communication style)
        void top_send(uint8_t value);   // buffer a value to be sent in a batch (BATCH communication style), or send it immediately (SINGLE communication style)
        void clean_top_channel();       // send the batch of values to the next process asynchronously (BATCH communication style), 
                                        // or do nothing (SINGLE communication style)
        void sync_top_channel();        // wait for the batch to be received by the next process (BATCH communication style), or do nothing (SINGLE communication style)
        void bot_receive();             // receive a value from the previous process bottom stream and buffer it into a queue (both communication styles)
        void bot_send(uint8_t value);   // send a value to the next process bottom stream immediately (both communication styles)

        // let the compiler optimize the code for the specified sort direction
        template<SortDirection direction>
        void input_process(); // first process (rank == 0) reads the input values from STDIN and sends them in an alternating fashion to the 2nd process
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
void PipelineMergeSort<communication_style>::check_top_status(int size)
{
    if (_top_status.MPI_ERROR != MPI_SUCCESS)
    {
        cerr << "Error: Process " << RANK << " failed to receive message from TOP input stream." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1); // error occurred during the data transfer
    }

    int actual_size;
    MPI_Get_count(&_top_status, MPI_BYTE, &actual_size);
    if (actual_size != size)
    {
        cerr << "Error: Process " << RANK << " received message of unexpected size (" << actual_size << ") from TOP input stream." << endl;
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

    int actual_size;
    MPI_Get_count(&_bot_status, MPI_BYTE, &actual_size);
    if (actual_size != 1)
    {
        cerr << "Error: Process " << RANK << " received message of unexpected size (" << actual_size << ") from BOT input stream." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1); // not enough values received
    }
}

template<CommunicationStyles communication_style>
void PipelineMergeSort<communication_style>::setup_top_channel()
{
    if (communication_style == CommunicationStyles::BATCH)
    {
        // receive the whole batch of values from the previous process, much more efficient than receiving values one by one
        MPI_Recv(_input_buffer, INPUT_SIZE, MPI_BYTE, RANK - 1, 0, TOP_COMM, &_top_status);
        check_top_status((int)INPUT_SIZE);
    }
    else if (communication_style == CommunicationStyles::SINGLE)
    {
        for (uint64_t i = 0; i < INPUT_SIZE; i++)
        {
            // receive values one by one, i.e. simulate using a stream/queue
            MPI_Recv(&_input_buffer[i], 1, MPI_BYTE, RANK - 1, 0, TOP_COMM, &_top_status);
            check_top_status(1);
        }
    }
    _read_idx = 0;
}

template<CommunicationStyles communication_style>
void PipelineMergeSort<communication_style>::top_send(uint8_t value)
{
    if (communication_style == CommunicationStyles::BATCH)
    {
        // fill the output buffer, i.e. create a batch of values to be sent at once
        _output_buffer[_write_idx++] = value;
    }
    else if (communication_style == CommunicationStyles::SINGLE)
    {
        // send values one by one, i.e. simulate using a stream/queue
        MPI_Send(&value, 1, MPI_BYTE, RANK + 1, 0, TOP_COMM);
    }
}

template<CommunicationStyles communication_style>
void PipelineMergeSort<communication_style>::clean_top_channel()
{
    if (communication_style == CommunicationStyles::BATCH)
    {
        // avoid sending values one after another, it is not efficient to send them at once as there is a lot more overhead caused by the MPI communication
        // for smaller batches (processes with lower ranks) the communication will be buffered, for larger batches (processes with higher ranks) the send
        // will be blocking, but the communication will be more efficient, as the initiation latency of the communication will be small in comparison to the
        // time needed to send the whole batch
        MPI_Isend(_output_buffer, OUTPUT_SIZE, MPI_BYTE, RANK + 1, 0, TOP_COMM, &_top_send_request);
        _write_idx = 0;
    }
}

template<CommunicationStyles communication_style>
void PipelineMergeSort<communication_style>::sync_top_channel()
{
    if (communication_style == CommunicationStyles::BATCH)
    {
        MPI_Wait(&_top_send_request, &_top_status);
        if (_top_status.MPI_ERROR != MPI_SUCCESS)
        {
            cerr << "Error: Process " << RANK << " failed to synchronize TOP input stream." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1); // error occurred during the data transfer
        }
    }
}

template<CommunicationStyles communication_style>
void PipelineMergeSort<communication_style>::bot_send(uint8_t value)
{
    MPI_Send(&value, 1, MPI_BYTE, RANK + 1, 0, BOT_COMM);
}

template<CommunicationStyles communication_style>
void PipelineMergeSort<communication_style>::bot_receive()
{
    uint8_t value;
    MPI_Recv(&value, 1, MPI_BYTE, RANK - 1, 0, BOT_COMM, &_bot_status);
    check_bot_status();
    _input_queue.push(value);
}

template<CommunicationStyles communication_style>
void PipelineMergeSort<communication_style>::sort(SortDirection direction)
{
    // decide the direction of the sorting in advance to be able to use compile-time templates for optimization
    if (direction == ASCENDING)
    {
        if (RANK == 0) // first process
        {
            input_process<ASCENDING>();
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
            input_process<DESCENDING>();
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
template<SortDirection direction>
void PipelineMergeSort<communication_style>::input_process()
{
    union
    {
        uint8_t value;
        char character;
    } character_value; // dirty trick to cast values between char and uint8_t

    uint64_t i = 0;
    for (; i < ITERATIONS; i++)
    {    
        // read the input value from stdin
        if (!cin.get(character_value.character))
        {
            break; // no more input values, padding will be used to fill the rest of the input values
        }
        
        if (_ping_pong) // write to top channel
        {
            MPI_Send(&character_value.value, 1, MPI_BYTE, 1, 0, TOP_COMM);
        }
        else // write to bottom stream
        {
            MPI_Send(&character_value.value, 1, MPI_BYTE, 1, 0, BOT_COMM);
        }

        _ping_pong = !_ping_pong; // switch the channel/stream
    }

    if (i < ITERATIONS >> 1) // there must be at least 2^(size-2) input values, otherwise the number of processes is too high
    {
        cerr << "Error: There are less input values present than expected." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1); // not enough input values
    }

    MPI_Request size_request;
    MPI_Isend(&i, 1, MPI_UINT64_T, SIZE-1, 0, MPI_COMM_WORLD, &size_request); // send the actual number of input values to the last process

    uint8_t padding = direction == ASCENDING ? 0xFF : 0x00; // padd the input values with the maximum (ASCENDING sort) or minimum (DESCENDING sort) value
    for (; i < ITERATIONS; i++)
    {
        if (_ping_pong) // write to top channel
        {
            MPI_Send(&padding, 1, MPI_BYTE, 1, 0, TOP_COMM);
        }
        else // write to bottom stream
        {
            MPI_Send(&padding, 1, MPI_BYTE, 1, 0, BOT_COMM);
        }

        _ping_pong = !_ping_pong; // switch the channel/stream
    }
    
    cin.get(character_value.character); // try to read one more value
    if (!cin.eof()) // there are still input values present
    {
        cerr << "Warning: There are still input values present, which will not be sorted." << endl;
    }

    MPI_Wait(&size_request, MPI_STATUS_IGNORE); // wait for the number of input values to be sent to the last process
}

template<CommunicationStyles communication_style> 
template<SortDirection direction>
void PipelineMergeSort<communication_style>::merge_process()
{
    for (uint64_t n = 0; n < ITERATIONS; n++)
    {
        setup_top_channel(); // receive 2^(rank-1) values from the previous process

        uint64_t i = 0, j = 0; // i - top channel index, j - bottom stream index
        if (_ping_pong) // write to top channel in batches
        {
            while (j++ < INPUT_SIZE) // read all values from the bottom stream as fast as possible and buffer them if necessary
            {
                bot_receive(); // receive a value from the bottom stream 

                // read/write from the top channel until the value from the bottom stream is smaller (ASCENDING sort) or greater (DESCENDING sort)
                if ((direction == ASCENDING && _input_buffer[i] < _input_queue.front()) || (direction == DESCENDING && _input_buffer[i] > _input_queue.front()))
                {
                    top_send(_input_buffer[i++]); // this can be buffered or sent immediately depending on the communication style
                }
                else
                {
                    top_send(_input_queue.front());
                    _input_queue.pop();
                }
            }

            while (!_input_queue.empty() && i < INPUT_SIZE) // empty one of the input data structures
            {
                // read/write from the top channel until the value from the bottom stream is smaller (ASCENDING sort) or greater (DESCENDING sort)
                if ((direction == ASCENDING && _input_buffer[i] < _input_queue.front()) || (direction == DESCENDING && _input_buffer[i] > _input_queue.front()))
                {
                    top_send(_input_buffer[i++]); // this can be buffered or sent immediately depending on the communication style
                }
                else
                {
                    top_send(_input_queue.front());
                    _input_queue.pop();
                }
            }

            // only one of the while loops will be executed, the other one will be skipped
            while (i < INPUT_SIZE) // read/write the remaining values from the top channel, if there are any left
            {
                top_send(_input_buffer[i++]);
            }

            while (!_input_queue.empty())
            {
                top_send(_input_queue.front());
                _input_queue.pop();
            }

            clean_top_channel(); // send the merge-sorted batch to the next process in case of the BATCH communication style
        }
        else // write to bottom stream, here it is not necessary to write values one after another, so the next process can start sorting
        {
            while (j++ < INPUT_SIZE) // read all values from the bottom stream as fast as possible and buffer them if necessary
            {
                bot_receive(); // receive a value from the bottom stream

                // send values from the top channel until the value from the bottom stream is smaller (ASCENDING sort) or greater (DESCENDING sort)
                if ((direction == ASCENDING && _input_buffer[i] < _input_queue.front()) || (direction == DESCENDING && _input_buffer[i] > _input_queue.front()))
                {
                    bot_send(_input_buffer[i++]); // send the value from the top channel, now it is smaller (ASCENDING sort) or greater (DESCENDING sort) 
                }
                else
                {
                    bot_send(_input_queue.front());
                    _input_queue.pop();
                }
            }

            while (!_input_queue.empty() && i < INPUT_SIZE) // empty one of the input data structures
            {
                // send values from the top channel until the value from the bottom stream is smaller (ASCENDING sort) or greater (DESCENDING sort)
                if ((direction == ASCENDING && _input_buffer[i] < _input_queue.front()) || (direction == DESCENDING && _input_buffer[i] > _input_queue.front()))
                {
                    bot_send(_input_buffer[i++]); // send the value from the top channel, now it is smaller (ASCENDING sort) or greater (DESCENDING sort) 
                }
                else
                {
                    bot_send(_input_queue.front());
                    _input_queue.pop();
                }
            }

            // only one of the while loops will be executed, the other one will be skipped
            while (i < INPUT_SIZE) // send the remaining values from the top channel, if there are any left to be sent
            {
                bot_send(_input_buffer[i++]);
            }

            while (!_input_queue.empty()) // send the remaining values from the bottom stream, if there are any left to be sent
            {
                bot_send(_input_queue.front());
                _input_queue.pop();
            }
        }

        _ping_pong = !_ping_pong; // switch the channel/stream

        if (communication_style == CommunicationStyles::BATCH && _ping_pong)
        {
            sync_top_channel(); // wait for the batch sent in a previous iteration to be received by the next process
        }
    }
}

template<CommunicationStyles communication_style> 
template<SortDirection direction>
void PipelineMergeSort<communication_style>::output_process()
{
    uint64_t outputs = 0;
    MPI_Request size_request;
    MPI_Irecv(&outputs, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &size_request); // receive the number of input values from the first process

    setup_top_channel(); // wait for half of the values to be received

    uint64_t i = 0, j = 0;
    // print half of the values sorted
    while (j++ < INPUT_SIZE) // read all values from the bottom stream as fast as possible and buffer them if necessary
    {
        bot_receive();
        if ((direction == ASCENDING && _input_buffer[i] < _input_queue.front()) || (direction == DESCENDING && _input_buffer[i] > _input_queue.front()))
        {
            cout << (unsigned)_input_buffer[i++] << endl;
        }
        else
        {
            cout << (unsigned)_input_queue.front() << endl;
            _input_queue.pop();
        }
    }

    uint64_t k = INPUT_SIZE;
    MPI_Wait(&size_request, MPI_STATUS_IGNORE); // wait for the number of actual input values to be received from the first process

    while (!_input_queue.empty() && i < INPUT_SIZE && k++ < outputs) // empty one of the input data structures or stop if there are no more values to be printed
    {
        if ((direction == ASCENDING && _input_buffer[i] < _input_queue.front()) || (direction == DESCENDING && _input_buffer[i] > _input_queue.front()))
        {
            cout << (unsigned)_input_buffer[i++] << endl;
        }
        else
        {
            cout << (unsigned)_input_queue.front() << endl;
            _input_queue.pop();
        }
    }

    // only one of the while loops will be executed, the other one will be skipped
    while (i < INPUT_SIZE && k++ < outputs) // print the remaining values from the top channel, if there are any left to be printed
    {
        cout << (unsigned)_input_buffer[i++] << endl;
    }

    while (!_input_queue.empty() && k++ < outputs) // print the remaining values from the bottom stream, if there are any left to be printed
    {
        cout << (unsigned)_input_queue.front() << endl;
        _input_queue.pop();
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

    // default values for the sort direction and the communication style
    SortDirection direction = SortDirection::ASCENDING;
    CommunicationStyles communication_style = CommunicationStyles::BATCH;

    // parsing of arguments, very simple, second argument can be passed only if the first argument is passed
    if (argc > 1) // parse the first command line argument (optional)
    {
        string arg = argv[1];
        if (arg == "-d")
        {
            direction = SortDirection::DESCENDING;
            INFO_PRINT(rank, "DESCENDING sort will be performed.");
        }
        else
        {
            INFO_PRINT(rank, "ASCENDING sort will be performed.");
        }
        
        if (rank == 0 && ((arg != "-a" && arg != "-d"))) // valid arguments are '-a' for ASCENDING sort and '-d' for DESCENDING sort
        {
            cerr << "Warning: Invalid 1st argument, expected '-a' for ASCENDING sort or '-d' for DESCENDING sort." << endl;
            cerr << "         ASCENDING sort will be performed by default." << endl;
        }
    }

    if (argc > 2) // parse the second command line argument (optional)
    {
        string arg = argv[2];
        if (arg == "-s")
        {
            communication_style = CommunicationStyles::SINGLE;
            INFO_PRINT(rank, "SINGLE communication style will be used.");
        }
        else
        {
            INFO_PRINT(rank, "BATCH communication style will be used.");
        }
        
        if (rank == 0 && ((arg != "-s" && arg != "-b")))
        {
            if (rank == 0) // valid arguments are '-s' for SINGLE communication style and '-b' for BATCH communication style
            {
                cerr << "Warning: Invalid 2nd argument, expected '-s' for SINGLE communication style or '-b' for BATCH communication style." << endl;
                cerr << "         BATCH communication style will be used by default." << endl;
            }
        }
    }

    MPI_Comm top_comm, bot_comm;
    // simulate the top channel with a separate communicator, the top channel will be always used to send a whole batch of data, 
    // i.e. 2^rank elements, since the rank+1 process can only start computing after the rank process has finished 2^rank elements
    MPI_Comm_dup(MPI_COMM_WORLD, &top_comm);
    // simulate the bottom channel with another separate communicator, the bottom channel will be always used to send values one by one
    MPI_Comm_dup(MPI_COMM_WORLD, &bot_comm);

    // let the compiler compile all the code for the specified communication style and sort direction
    if (communication_style == CommunicationStyles::BATCH)
    {
        PipelineMergeSort<CommunicationStyles::BATCH> pms(rank, size, top_comm, bot_comm); // initialize the pipeline sorter
        pms.sort(direction);                                                               // perform the sorting
    }
    else if (communication_style == CommunicationStyles::SINGLE)
    {
        PipelineMergeSort<CommunicationStyles::SINGLE> pms(rank, size, top_comm, bot_comm); // initialize the pipeline sorter
        pms.sort(direction);                                                                // perform the sorting
    }

    // wait for all processes to finish
    MPI_Barrier(MPI_COMM_WORLD);

    // clean up
    MPI_Comm_free(&top_comm);
    MPI_Comm_free(&bot_comm);
    MPI_Finalize();

    return 0;
}
