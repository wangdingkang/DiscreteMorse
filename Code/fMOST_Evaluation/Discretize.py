import os
import sys
from math import sqrt
import argparse

# input_swc = 'RESULT_01_15_FMOST/neuron2_smaller/neuron2_smaller_gtree.swc'
# output_swc = 'RESULT_01_15_FMOST/neuron2_smaller/neuron2_gtree_discrete.swc'

parser = argparse.ArgumentParser()
parser.add_argument('input_path', help=r"Path to the target swc file.")
parser.add_argument('output_path', help=r"Path to the output swc file.")
args = parser.parse_args()

input_swc = args.input_path
output_swc = args.output_path

if __name__ == '__main__':
    nodes = {}
    new_nodes = []
    old_nodes_str = []

    with open(input_swc) as file:
        for line in file:
            if not line.startswith('#'):
                old_nodes_str.append(line)
                data = line.strip().split()
                id, pid = data[0], data[6]
                x, y, z = map(float, data[2:5])
                nodes[id] = (x, y, z)

    with open(input_swc) as file:
        for line in file:
            if not line.startswith('#'):
                data = line.strip().split()
                id, pid = data[0], data[6]
                x, y, z = map(float, data[2:5])
                if pid != '-1':
                    px, py, pz = nodes[pid]
                    dist = int(sqrt((x-px)**2 + (y-py)**2 + (z-pz)**2))
                    if dist > 1:
                        dx, dy, dz = (px-x)/dist, (py-y)/dist, (pz-z)/dist
                        for r in range(1, dist):
                            nx, ny, nz = x + dx * r, y + dy *r, z + dz*r
                            new_nodes.append((nx, ny, nz))
        for n in new_nodes:
            old_nodes_str.append('1 1 {} {} {} 1 -1\n'.format(n[0], n[1], n[2]))
        print(len(new_nodes), 'node added.')
        with open(output_swc, 'w') as file:
            for line in old_nodes_str:
                file.write(line)