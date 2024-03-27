h5dump -w 10000 temp_par.h5 >debug_par.txt 
h5dump -w 10000 temp_seq.h5 >debug_seq.txt 
sed -i 's/\(\.[0-9]\+\),/,/g' debug_seq.txt
sed -i 's/\(\.[0-9]\+\),/,/g' debug_par.txt
git --no-pager diff --word-diff=porcelain --color-words=. --no-index -U0 debug_par.txt debug_seq.txt
