
make test

for processes in 2 4 8; do
    for dir in ./solved_grids/*; do
        echo "Testing $dir"
        file_count=$(ls -1 $dir | wc -l)
        for i in $(seq 0 $(($file_count - 1))); do
            for j in $(seq $i $(($file_count - 1))); do
                iterations=$(($file_count - $j))
                padded_j=$j
                if [ $j -lt 10 ]; then
                    padded_j="0$j"
                fi

                for k in $(seq 1 $(($iterations - 1))); do
                    echo -e "\e[34mTesting $dir/$padded_j.txt for $k iterations\e[0m"
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
    done
    echo -e "\e[32mTests passed for $processes processes\e[0m"    
done

echo -e "\e[32mALL TESTS PASSED\e[0m"