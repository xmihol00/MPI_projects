ml matplotlib
ml OpenMPI
mpic++ -O3 -o pms ../pms.cpp

mkdir -p performance
python3 -c "import numpy as np; np.random.randint(0, 256, 2**32, dtype=np.uint8).tofile('nums.bin')"

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
        rm -f performance/$file_name
        for N in {2..26}; do
            M=$((2**($N-1)))
            echo "N=$N, M=$M, D=$D, C=$C"
            start_time=$(date +%s.%N)
            mpiexec -np $N pms $D $C <nums.bin >/dev/null
            end_time=$(date +%s.%N)
            diff=$(echo "scale=3; $end_time - $start_time" | bc)
            echo "$N, $diff" >> performance/$file_name
        done
    done
done
