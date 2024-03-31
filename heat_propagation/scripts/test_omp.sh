tile_sizes=(512 2048)
processes=(2 4 8 16)
iterations=(251 444)
threads=(8 4 2)
ems=(1 2)

make
echo "" | tee results.txt

for tile_size in "${tile_sizes[@]}"; do
    ./data_generator -n $tile_size
    for process in "${processes[@]}"; do
        if [ $process -lt $tile_size ]; then
            for em in "${ems[@]}"; do
                for iteration in "${iterations[@]}"; do
                    for thread in "${threads[@]}"; do
                        echo "Tile size: $tile_size, Processes: $process, Iterations: $iteration, Threads: $thread" | tee -a results.txt
                        
                        if [ $((2*$process)) -lt $tile_size ]; then
                            echo "command -s $tile_size: mpirun --oversubscribe -np $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -t $thread -v -o temp" | tee -a results.txt
                            mpirun --oversubscribe -np $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -t $thread -v -o temp | tee -a results.txt
                            ../heat_propagation/scripts/h5_comparrison.sh | tee -a results.txt
                            echo "" | tee -a results.txt
                            echo "command -s $tile_size: mpirun --oversubscribe -np $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -t $thread -v -o temp -p" | tee -a results.txt
                            mpirun --oversubscribe -np $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -t $thread -v -o temp -p | tee -a results.txt
                            ../heat_propagation/scripts/h5_comparrison.sh | tee -a results.txt
                            echo "" | tee -a results.txt
                        fi

                        echo "command -s $tile_size: mpirun --oversubscribe -np $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -t $thread -v -g -o temp" | tee -a results.txt
                        mpirun --oversubscribe -np $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -t $thread -v -g -o temp | tee -a results.txt
                        ../heat_propagation/scripts/h5_comparrison.sh | tee -a results.txt
                        echo "" | tee -a results.txt
                        echo "command -s $tile_size: mpirun --oversubscribe -np $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -t $thread -v -g -o temp -p" | tee -a results.txt
                        mpirun --oversubscribe -np $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -t $thread -v -g -o temp -p | tee -a results.txt
                        ../heat_propagation/scripts/h5_comparrison.sh | tee -a results.txt
                        echo "" | tee -a results.txt
                    done
                done
            done
        fi
    done
done

rm temp_par.h5
rm temp_seq.h5
