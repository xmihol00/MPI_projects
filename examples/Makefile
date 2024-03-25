CC = mpic++

.PHONY: rcs ccs tcs rd cd td cs ts ss

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

cs:
	$(CC) -o col_scatter col_scatter.cpp
	mpirun -np 5 --oversubscribe ./col_scatter

ts:
	$(CC) -o tile_scatter tile_scatter.cpp
	mpirun -np 4 --oversubscribe ./tile_scatter

ss:
	$(CC) -o struct_scatter struct_scatter.cpp
	mpirun -np 4 --oversubscribe ./struct_scatter

gcs:
	$(CC) -o group_comm_split group_comm_split.cpp
	mpirun -np $N --oversubscribe ./group_comm_split

srp:
	$(CC) -o scan_reduce_pack scan_reduce_pack.cpp
	mpirun -np $N --oversubscribe ./scan_reduce_pack
