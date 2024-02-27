dir="performance_NTB"
mkdir -p $dir


for D in -a -d; do
    for C in -b -s; do
        if [ "$D" = "-a" ]; then
            file_name="ascending"
            R=""
        else
            file_name="descending"
            R="-r"
        fi
        if [ "$C" = "-b" ]; then
            file_name="${file_name}_batch.csv"
        else
            file_name="${file_name}_single.csv"
        fi
        echo -e "testing: \e[34m$file_name\e[0m"
        rm -f $dir/$file_name
        for N in {2..26}; do
            M=$((2**($N-1)))
            echo "N=$N, M=$M, D=$D, C=$C"
            start_time=$(date +%s.%N)
            head -c $M ../nums.bin | mpirun --oversubscribe -np $N pms $D $C >/dev/null
            end_time=$(date +%s.%N)
            diff=$(echo "scale=3; $end_time - $start_time" | bc)
            echo "$N, $diff" >> $dir/$file_name
        done
    done
done
