Description of the solution:
1. Processes are organized into a 2D (cartesian) mesh topology. The dimensions of the mesh can be specified by the user, see TODO, or are derived 
   automatically. The automatically derived mesh dimensions will always contain a power of 2 processes, see TODO. Processes that do not fit into the 
   mesh will not be utilized. Lastly, each process in the mesh retrieves ranks of its neighbors, especially of its corner neighbors (NW, NE, SE, SW).
2. The input file is read only by the root process. Based on the size of the input file and the length of the first row, the global grid dimensions, 
   i.e. the simulation space, are determined. The global grid dimensions are adjusted to be divisible by the number of nodes in each dimension of the
   mesh. The content of the file is then read to the global grid with evenly distributed padding of '0' if necessary or as specified by the user.
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