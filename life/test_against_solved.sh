
make test

for processes in 2 4 8; do
    for dir in ./solved_grids/*; do
        echo "Testing $dir"
        file_count=$(ls -1 $dir | wc -l)
        if [ $file_count -lt 10 ]; then # test each file as the starting point and verify all the next possible iterations from that file
            for i in $(seq 0 $(($file_count - 1))); do
                for j in $(seq $i $(($file_count - 1))); do
                    iterations=$(($file_count - $j))
                    padded_j=$j
                    if [ $j -lt 10 ]; then
                        padded_j="0$j"
                    fi

                    for k in $(seq 1 $(($iterations - 1))); do
                        echo -e "\e[34mTesting $dir/$padded_j.txt for $k iterations with $processes processes\e[0m"
                        # the result must be printed to stderr without any additional output, just the grid
                        mpiexec --oversubscribe -n $processes ./life_test $dir/$padded_j.txt $k 2>tmp.out
                        
                        reference=$((j + k))
                        if [ $reference -lt 10 ]; then
                            reference="0$reference"
                        fi

                        diff -s -Z tmp.out $dir/$reference.txt
                        if [ $? -ne 0 ]; then
                            echo -e "\e[31mTEST FAILED\e[0m"
                            exit 1
                        fi
                        echo ""
                    done
                done
            done
        else # test just the starting file and verify all the next possible iterations from that file
            for i in $(seq 0 $(($file_count - 1))); do
                padded_i=$i
                if [ $i -lt 10 ]; then
                    padded_i="0$i"
                fi
                echo -e "\e[34mTesting $dir/00.txt for $i iterations with $processes processes\e[0m"
                mpiexec --oversubscribe -n $processes ./life_test $dir/00.txt $i 2>tmp.out

                diff -s -Z tmp.out $dir/$padded_i.txt
                if [ $? -ne 0 ]; then
                    echo -e "\e[31mTEST FAILED\e[0m"
                    exit 1
                fi
                echo ""
            done
        fi
    done
    echo -e "\e[32mTests passed for $processes processes\e[0m"    
done

echo -e "\e[32mALL TESTS PASSED\e[0m"