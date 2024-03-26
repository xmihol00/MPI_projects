tile_sizes=(64 128 256)
processes=(2 4 8 16 32)
iterations=(125 250 444)
ems=(1 2)

make
echo "" | tee results.txt

for tile_size in "${tile_sizes[@]}"; do
    ./data_generator -n $tile_size
    for process in "${processes[@]}"; do
        if [ $process -lt $tile_size ]; then
            for em in "${ems[@]}"; do
                for iteration in "${iterations[@]}"; do
                    echo "Tile size: $tile_size, Processes: $process, Iterations: $iteration" | tee -a results.txt
                    
                    if [ $((2*$process)) -lt $tile_size ]; then
                        echo "command -s $tile_size: mpirun --oversubscribe -np $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -v -o temp" | tee -a results.txt
                        mpirun --oversubscribe -np $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iterations -v -o temp | tee -a results.txt
                        ../scripts/h5_comparrison.sh | tee -a results.txt
                        echo "" | tee -a results.txt
                        echo "command -s $tile_size: mpirun --oversubscribe -np $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -v -o temp -p" | tee -a results.txt
                        mpirun --oversubscribe -np $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iterations -v -o temp -p | tee -a results.txt
                        ../scripts/h5_comparrison.sh | tee -a results.txt
                        echo "" | tee -a results.txt
                    fi

                    echo "command -s $tile_size: mpirun --oversubscribe -np $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -v -g -o temp" | tee -a results.txt
                    mpirun --oversubscribe -np $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iterations -v -g -o temp | tee -a results.txt
                    ../scripts/h5_comparrison.sh | tee -a results.txt
                    echo "" | tee -a results.txt
                    echo "command -s $tile_size: mpirun --oversubscribe -np $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -v -g -o temp -p" | tee -a results.txt
                    mpirun --oversubscribe -np $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iterations -v -g -o temp -p | tee -a results.txt
                    ../scripts/h5_comparrison.sh | tee -a results.txt
                    echo "" | tee -a results.txt
                done
            done
        fi
    done
done