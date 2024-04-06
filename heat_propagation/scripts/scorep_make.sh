source load_modules.sh
mkdir -p ../build_prof
cd ../build_prof
rm -rf *
CPP_SOURCE_DIR=../sources
/apps/all/Score-P/8.0-iimpi-2021b/bin/scorep mpiicpc -DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX -D_MPICC_H -I../3rdparty -isystem /apps/all/HDF5/1.12.1-intel-2021b-parallel/include -isystem /apps/all/zlib/1.2.11-GCCcore-11.2.0/include -isystem /apps/all/Szip/2.1.1-GCCcore-11.2.0/include -isystem /apps/all/imkl/2021.4.0/mkl/2021.4.0/include -Wall -Xlinker -lstdc++ -ipo -O3 -no-prec-div -fp-model=fast=2 -xHost -qopenmp -std=gnu++17  $CPP_SOURCE_DIR/main.cpp $CPP_SOURCE_DIR/HeatSolverBase.cpp $CPP_SOURCE_DIR/MaterialProperties.cpp $CPP_SOURCE_DIR/ParallelHeatSolver.cpp $CPP_SOURCE_DIR/SequentialHeatSolver.cpp $CPP_SOURCE_DIR/SimulationProperties.cpp $CPP_SOURCE_DIR/utils.cpp /apps/all/HDF5/1.12.1-intel-2021b-parallel/lib/libhdf5.so /apps/all/Szip/2.1.1-GCCcore-11.2.0/lib/libsz.so /apps/all/zlib/1.2.11-GCCcore-11.2.0/lib/libz.so /usr/lib64/libdl.so -lm /apps/all/intel-compilers/2021.4.0/compiler/2021.4.0/linux/compiler/lib/intel64/libiomp5.so /usr/lib64/libpthread.so /apps/all/imkl/2021.4.0/compiler/2021.4.0/linux/compiler/lib/intel64_lin/libiomp5.so -o ppp_proj01
