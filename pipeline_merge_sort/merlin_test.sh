#!/bin/bash
# scp pms.cpp xmihol00@eva.fit.vutbr.cz:PRL/
# scp merlin_test.sh xmihol00@eva.fit.vutbr.cz:PRL/

mpic++ --prefix /usr/local/share/OpenMPI -o pms pms.cpp
echo $(date +"%Y-%m-%d_%H-%M-%S") >> test_out.txt

for numbers in 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 30 31 32 33 34 62 63 64 65 66 127 128 129 255 256 257 511 512 513 1023 1024 1025 2047 2048 100 111 200 222 300 333 400 444 500 555 600 666 700 777 800 888 900 999 1000 1111 2000; do
    calc=$(echo "(l($numbers)/l(2))+1" | bc -l)
    proc=$(python3 -c "from math import ceil; print (ceil($calc))")

    echo "N=$proc, M=$numbers" | tee -a test_out.txt
    dd if=/dev/random bs=1 count=$numbers 2>/dev/null | mpirun --prefix /usr/local/share/OpenMPI  -np $proc pms | tee out.txt | (sort -nc && echo -e "\e[32mSORTED\e[0m" || echo -e "\e[31mNOT SORTED\e[0m") | tee -a test_out.txt
    if [ $(wc -l < out.txt) -eq $numbers ]; then
        echo -e "\e[32mCORRECT COUNT\e[0m" | tee -a test_out.txt
    else
        echo -e "\e[31mINCORRECT COUNT\e[0m" | tee -a test_out.txt
    fi
done

rm -f pms out.txt

# cat test_out.txt | grep -E "INCORRECT|NOT SORTED"
