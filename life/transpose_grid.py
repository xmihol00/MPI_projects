import argparse
import numpy as np
import glob
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('in_dir', type=str)
argparser.add_argument('out_dir', type=str)

args = argparser.parse_args()

for filename in glob.glob(os.path.join(args.in_dir, '*')):
    basename = os.path.basename(filename)
    print("transposing:", basename)
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    grid = np.array([list(line.strip()) for line in lines])
    grid = list(grid.T)
    
    with open(os.path.join(args.out_dir, basename), 'w') as f:
        for line in grid:
            f.write(''.join(line))
            f.write('\n')
