oversubscribe="--oversubscribe"
iterations=20
if pwd | grep "xmihol00"; then
    module add openmpi-4.0.3-gcc
    oversubscribe=""
    iterations=32
fi

echo -e "\e[34mAscending batch\e[0m"
D="-a"
C="-b"
for ((N=2; N<=$iterations; N++)); do
    M=$((2**($N-1)))
    echo "N=$N, M=$M"
    if [ "$D" = "-d" ]; then R="-r"; else R=""; fi
    dd if=/dev/random bs=1 count=$M 2>/dev/null | mpirun --use-hwthread-cpus $oversubscribe -np $N ./pms $D $C | sort -nc $R && echo -e "\e[32mSORTED\e[0m" || echo -e "\e[31mNOT SORTED\e[0m"
done

echo ""
echo -e "\e[34mDescending batch\e[0m"
D="-d"
C="-b"
for ((N=2; N<=$iterations; N++)); do
    M=$((2**($N-1)))
    echo "N=$N, M=$M"
    if [ "$D" = "-d" ]; then R="-r"; else R=""; fi
    dd if=/dev/random bs=1 count=$M 2>/dev/null | mpirun --use-hwthread-cpus $oversubscribe -np $N ./pms $D $C | sort -nc $R && echo -e "\e[32mSORTED\e[0m" || echo -e "\e[31mNOT SORTED\e[0m"
done

echo ""
echo -e "\e[34mAscending single\e[0m"
D="-a"
C="-s"
for ((N=2; N<=$iterations; N++)); do
    M=$((2**($N-1)))
    echo "N=$N, M=$M"
    if [ "$D" = "-d" ]; then R="-r"; else R=""; fi
    dd if=/dev/random bs=1 count=$M 2>/dev/null | mpirun --use-hwthread-cpus $oversubscribe -np $N ./pms $D $C | sort -nc $R && echo -e "\e[32mSORTED\e[0m" || echo -e "\e[31mNOT SORTED\e[0m"
done

echo ""
echo -e "\e[34mDescending single\e[0m"
D="-d"
C="-s"
for ((N=2; N<=$iterations; N++)); do
    M=$((2**($N-1)))
    echo "N=$N, M=$M"
    if [ "$D" = "-d" ]; then R="-r"; else R=""; fi
    dd if=/dev/random bs=1 count=$M 2>/dev/null | mpirun --use-hwthread-cpus $oversubscribe -np $N ./pms $D $C | sort -nc $R && echo -e "\e[32mSORTED\e[0m" || echo -e "\e[31mNOT SORTED\e[0m"
done
