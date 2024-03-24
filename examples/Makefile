CC = mpic++

.PHONY: rcs rcs_run ccs ccs_run tcs tcs_run

ifndef N
N=16
endif

rcs:
	$(CC) -o row_comm_split row_comm_split.cpp

ccs:
	$(CC) -o col_comm_split col_comm_split.cpp

tcs:
	$(CC) -o tile_comm_split tile_comm_split.cpp

rcs_run: rcs
	mpirun -np $N --oversubscribe ./row_comm_split

ccs_run: ccs
	mpirun -np $N --oversubscribe ./col_comm_split

tcs_run: tcs
	mpirun -np $N --oversubscribe ./tile_comm_split
