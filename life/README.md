# John Conway's Game Of Life Simulation
Multi-process simulation of the John Conway's Game of Life implemented with the MPI library to the Parallel and Distributed Algorithms course at FIT, BUT.

## Description Of The Solution
1. Processes are organized into a 2D (cartesian) mesh topology. The dimensions of the mesh can be specified by the user, or are derived automatically. 
   The automatically derived mesh dimensions will always contain a power of 2 processes. Processes that do not fit into the mesh will not be utilized. 
   Lastly, each process in the mesh retrieves ranks of its neighbors, especially of its corner neighbors (NW, NE, SE, SW).
2. The input file is read only by the root process. Based on the size of the input file and the length of the first row, the global grid dimensions, 
   i.e. the space of the simulation, are determined. The global grid dimensions are adjusted to be divisible by the number of nodes in each dimension 
   of the mesh. The content of the file is then placed to the left upper corner of the global grid. The rest is padded with '0'.
3. The root process computes the sizes of local tiles, i.e. the parts of the global grid assigned to each process, and broadcasts this information to
   all the processes.
4. Each process creates MPI data types specifying its local tile, the tile with halo zones (edges of the local tiles of neighboring processes, which 
   must be visible to a given process), and the halo zones. These MPI data types are same for all processes. 
5. Each process allocates memory for two local tiles, one for the current state of the simulation and one for the next state to realize so called 
   ping-pong buffering.
6. The root process partitions (not explicitly, this is done by MPI) the global grid into local tiles and scatters them to all processes in the mesh.
   Initial halo zone exchange is performed, i.e. each process sends its halo zones to its neighbors and receives halo zones from its neighbors. Now, 
   all processes have the initial state, which is also copied to the second local tile.
7. The simulation is started. In each iteration, the following steps are performed:
   * Each process computes the next state around the edges of its local tile, i.e. the halo zones.
   * Non-blocking exchange of halo zones is initiated.
   * The computation of the next state is performed (tile edges not included anymore). This is accelerated with SIMD instructions.
   * The exchange of halo zones is awaited.
   * current and next state tiles are swapped.
8. After the simulation is finished, the local tiles are gathered back to the root process (without the halo zones) and the content of the global grid
   is printed in a table-like format which shows what part of the grid was computed by which process.

See more details in the `life.cpp` file.

## Usage
Compile with the `make` command.
```
mpiexec -np <number of processes> ./life <grid file name> <number of iterations> [options]
Options:
  -w,   --wraparound                Use wrapped around simulation.
  -nx,  --nodes_x <number>          Number of nodes (processes) in the X direction of the mesh topology.
  -ny,  --nodes_y <number>          Number of nodes (processes) in the Y direction of the mesh topology.
  -p,   --padding <number>          Padding of the global grid in all directions.
  -px,  --padding_x <number>        Padding of the global grid in the X (height) direction.
  -py,  --padding_y <number>        Padding of the global grid in the Y (width) direction.
  -pt,  --padding_top <number>      Padding of the global grid from the top.
  -pb,  --padding_bottom <number>   Padding of the global grid from the bottom.
  -pl,  --padding_left <number>     Padding of the global grid from the left.
  -pr,  --padding_right <number>    Padding of the global grid from the right.
  -ppc, --pixels_per_cell <number>  Number of pixels per cell in the generated images/video.
  -iod, --images_output_directory <directory>
                                    Directory where generated images will be stored to.
  -v,   --video <file name>         Generate a video of the simulation in mp4 format.
  -fps, --frames_per_second <number>
                                    Frames per second in the generated video.
  -nfp, --no_formatted_print        Do not print the global grid in a table-like format.
  -ep,  --stderr_print              Print the unformatted global grid to stderr.
  -h,   --help                      Print this help message.
```

Examples:
* `mpiexec -n 4 ./life other_grids/glider_create_gun.txt 600 -v glider_gun.mp4 -fps 10 -pb 9 -pr 3`
* `mpiexec --oversubscribe -n 6 ./life wraparound_solved_grids/glider_8x8/00.txt 900 -w -v glider.mp4 -p 10 -fps 15 -ny 3`
* ```
  mpiexec --oversubscribe -n 64 ./life wraparound_solved_grids/glider_8x8/00.txt 11 -w 
    - extreme case with number of cells equal to the number of processes, output:
     |0|1|2|3|4|5|6|7|
    ------------------
    0|1|0|0|0|0|0|0|0|
    ------------------
    1|0|1|1|0|0|0|0|0|
    ------------------
    2|1|1|0|0|0|0|0|0|
    ------------------
    3|0|0|0|0|0|0|0|0|
    ------------------
    4|0|0|0|0|0|0|0|0|
    ------------------
    5|0|0|0|0|0|0|0|0|
    ------------------
    6|0|0|0|0|0|0|0|0|
    ------------------
    7|0|0|0|0|0|0|0|0|
    ------------------
```
