oversubscribe="--oversubscribe"
iterations=20
if pwd | grep "xmihol00"; then
    module add openmpi-4.0.3-gcc
    oversubscribe=""
    iterations=24
fi

mkdir -p performance

echo -e "\e[34mAscending batch\e[0m"
D="-a"
C="-b"
rm -f performance/ascending_batch.csv
for ((N=2; N<=$iterations; N++)); do
    M=$((2**($N-1)))
    echo "N=$N, M=$M"
    start_time=$(date +%s.%N)
    cat nums.bin | mpirun -np $N ./pms $D $C >/dev/null
    end_time=$(date +%s.%N)
    diff=$(echo "scale=3; $end_time - $start_time" | bc)
    echo "$N, $diff" >> performance/ascending_batch.csv
done

echo -e "\e[34mDescending batch\e[0m"
D="-d"
C="-b"
rm -f performance/descending_batch.csv
for ((N=2; N<=$iterations; N++)); do
    M=$((2**($N-1)))
    echo "N=$N, M=$M"
    start_time=$(date +%s.%N)
    cat nums.bin | mpirun  -np $N ./pms $D $C >/dev/null
    end_time=$(date +%s.%N)
    diff=$(echo "scale=3; $end_time - $start_time" | bc)
    echo "$N, $diff" >> performance/descending_batch.csv
done

echo -e "\e[34mAscending single\e[0m"
D="-a"
C="-s"
rm -f performance/ascending_single.csv
for ((N=2; N<=$iterations; N++)); do
    M=$((2**($N-1)))
    echo "N=$N, M=$M"
    start_time=$(date +%s.%N)
    cat nums.bin | mpirun  -np $N ./pms $D $C >/dev/null
    end_time=$(date +%s.%N)
    diff=$(echo "scale=3; $end_time - $start_time" | bc)
    echo "$N, $diff" >> performance/ascending_single.csv
done

echo -e "\e[34mDescending single\e[0m"
D="-d"
C="-s"
rm -f performance/descending_single.csv
for ((N=2; N<=$iterations; N++)); do
    M=$((2**($N-1)))
    echo "N=$N, M=$M"
    start_time=$(date +%s.%N)
    cat nums.bin | mpirun  -np $N ./pms $D $C >/dev/null
    end_time=$(date +%s.%N)
    diff=$(echo "scale=3; $end_time - $start_time" | bc)
    echo "$N, $diff" >> performance/descending_single.csv
done
