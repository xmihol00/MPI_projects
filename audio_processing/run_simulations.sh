
dealys=(33100 33200 33300 33400 33500 33600 33700)
jitters=(0 500 1000 2000 3000 5000)
buffers=(2 4 6 8)

for delay in "${dealys[@]}"; do
    for jitter in "${jitters[@]}"; do
        for buffer in "${buffers[@]}"; do
            echo "Running simulation with delay: $delay, jitter: $jitter, buffer: $buffer"
            mpiexec -np 2 ./rtap -i test_recording.wav -b $buffer -d $delay -j $jitter -o "b${buffer}_d${delay}_j${jitter}.wav"
        done
    done
done

