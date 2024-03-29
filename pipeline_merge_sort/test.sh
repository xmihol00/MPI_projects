now=$(date +"%Y-%m-%d_%H-%M-%S")
echo "$now" > failed.log
failed=0
for D in -a -d; do
    for C in -b -s; do
        if [ "$D" = "-a" ]; then
            title="Ascending"
        else
            title="Descending"
        fi
        if [ "$C" = "-b" ]; then
            title="$title Batch"
        else
            title="$title Bingle"
        fi
        echo -e "testing: \e[34m$title\e[0m" | tee -a failed.log
        for M in $(python -c "import numpy as np; samples = np.random.lognormal(mean=np.log(2**8), sigma=6, size=10000).astype(np.int64); print(' '.join(map(str, samples[(samples <= 2**20) & (samples >= 2)][:50])))"); do
            Q=$(echo "(l($M)/l(2))+1" | bc -l)
            N=$(python3 -c "from math import ceil; print(ceil($Q), end='')")
            echo "N=$N, M=$M, D=$D, C=$C"
            dd if=/dev/random bs=1 count=$M 2>/dev/null | mpirun --oversubscribe -np $N pms $D $C | python3 sorted.py $D $M >/dev/null
            if [ $? -eq 0 ]; then
                echo -e "\e[32mSORTED\e[0m"
            else
                echo "N=$N, M=$M, D=$D, C=$C" >> failed.log
                echo -e "\e[31mNOT SORTED\e[0m" | tee -a failed.log
            fi
        done
    done
done

echo "" | tee -a failed.log

for D in -a -d; do
    for C in -b -s; do
        if [ "$D" = "-a" ]; then
            title="Ascending"
            R=""
        else
            title="Descending"
            R="-r"
        fi
        if [ "$C" = "-b" ]; then
            title="$title Batch"
        else
            title="$title Bingle"
        fi
        echo -e "testing: \e[34m$title\e[0m" | tee -a failed.log
        for N in {2..20}; do
            M=$((2**($N-1)))
            echo "N=$N, M=$M, D=$D, C=$C"
            dd if=/dev/random bs=1 count=$M 2>/dev/null | mpirun --oversubscribe -np $N pms $D $C | python3 sorted.py $D $M >/dev/null
            if [ $? -eq 0 ]; then
                echo -e "\e[32mSORTED\e[0m"
            else
                echo "N=$N, M=$M, D=$D, C=$C" >> failed.log
                echo -e "\e[31mNOT SORTED\e[0m" | tee -a failed.log
            fi
        done
    done
done

if [ $failed -eq 0 ]; then
    echo "" | tee -a failed.log
    echo -e "\e[32mALL TESTS PASSED\e[0m" | tee -a failed.log
else
    echo -e "\e[31mSome tests failed\e[0m" | tee -a failed.log
fi
