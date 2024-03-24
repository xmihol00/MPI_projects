CC = mpic++

.PHONY: rcs ccs tcs rd cd td

ifndef N
N=16
endif

rcs:
	$(CC) -o row_comm_split row_comm_split.cpp
	mpirun -np $N --oversubscribe ./row_comm_split

ccs:
	$(CC) -o col_comm_split col_comm_split.cpp
	mpirun -np $N --oversubscribe ./col_comm_split

tcs:
	$(CC) -o tile_comm_split tile_comm_split.cpp
	mpirun -np $N --oversubscribe ./tile_comm_split

rd:
	$(CC) -o row_datatype row_datatype.cpp
	mpirun -np 2 --oversubscribe ./row_datatype

cd:
	$(CC) -o col_datatype col_datatype.cpp
	mpirun -np 2 --oversubscribe ./col_datatype

td:
	$(CC) -o tile_datatype tile_datatype.cpp
	mpirun -np 2 --oversubscribe ./tile_datatype