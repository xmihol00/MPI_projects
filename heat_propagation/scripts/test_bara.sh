tile_sizes=(256 1024)
processes=(2 4 8 16 32)
iterations=(251 444 511)
ems=(1 2)

source ../scripts/load_modules.sh
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
                        echo "command -s $tile_size: srun -N 1 -n $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -v -o temp" | tee -a results.txt
                        srun -N 1 -n $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -v -o temp | tee -a results.txt
                        ../scripts/h5_comparrison.sh | tee -a results.txt
                        echo "" | tee -a results.txt
                        echo "command -s $tile_size: srun -N 1 -n $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -v -o temp -p" | tee -a results.txt
                        srun -N 1 -n $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -v -o temp -p | tee -a results.txt
                        ../scripts/h5_comparrison.sh | tee -a results.txt
                        echo "" | tee -a results.txt
                    fi

                    echo "command -s $tile_size: srun -N 1 -n $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -v -g -o temp" | tee -a results.txt
                    srun -N 1 -n $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -v -g -o temp | tee -a results.txt
                    ../scripts/h5_comparrison.sh | tee -a results.txt
                    echo "" | tee -a results.txt
                    echo "command -s $tile_size: srun -N 1 -n $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -v -g -o temp -p" | tee -a results.txt
                    srun -N 1 -n $process ./ppp_proj01 -i ppp_input_data.h5 -m $em -n $iteration -v -g -o temp -p | tee -a results.txt
                    ../scripts/h5_comparrison.sh | tee -a results.txt
                    echo "" | tee -a results.txt
                done
            done
        fi
    done
done

(grep "FAILED" results.txt > /dev/null && echo -e "\n\033[0;31mSome tests FAILED.\033[0m" || echo -e "\n\033[0;34mAll tests PASSED.\033[0m") | tee -a results.txt

rm temp_par.h5
rm temp_seq.h5
rm temp_seq_int.txt
rm temp_par_int.txt
rm temp_seq.txt
rm temp_par.txt
