h5dump -w 10000 temp_par.h5 >temp_par.txt 
h5dump -w 10000 temp_seq.h5 >temp_seq.txt 
# remove all decimal values and keep just the integral parts to eliminate false positives due to floating point errors
sed 's/\(\.[0-9]\+\),/,/g' temp_seq.txt >temp_seq_int.txt
sed 's/\(\.[0-9]\+\),/,/g' temp_par.txt >temp_par_int.txt
git --no-pager diff --word-diff=porcelain --color-words=. --no-index -U0 temp_par_int.txt temp_seq_int.txt
