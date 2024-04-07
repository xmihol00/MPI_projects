
make test

for processes in 2 4 8 16; do
    for dir in ./wraparound_solved_grids/*; do
        echo "Testing $dir"
        file_count=$(ls -1 $dir | wc -l)
        for i in $(seq 0 $(($file_count - 1))); do
            padded_i=$i
            if [ $i -lt 10 ]; then
                padded_i="0$i"
            fi
            echo -e "\e[34mTesting $dir/00.txt for $i iterations with $processes processes\e[0m"
            mpiexec --oversubscribe -n $processes ./life_test $dir/00.txt $i -w 2>tmp.out

            diff -s -Z tmp.out $dir/$padded_i.txt
            if [ $? -ne 0 ]; then
                echo -e "\e[31mTEST FAILED\e[0m"
                exit 1
            fi
            echo ""
        done
    done
    echo -e "\e[32mTests passed for $processes processes\e[0m"    
done

echo -e "\e[32mALL TESTS PASSED\e[0m"