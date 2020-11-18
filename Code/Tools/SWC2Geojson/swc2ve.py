import os
import sys

input_swc = sys.argv[1]
output_vert = './DM_vert.txt'
output_edge = './DM_edge.txt'

vert = []
edge = []
with open(input_swc, 'r') as file:
    for line in file:
        t = line.strip().split()
        vert.append(t[2:5])
        id, pid = int(t[0]), int(t[-1])
        if pid != -1:
            edge.append([pid, id])


with open(output_vert, 'w') as file:
    for v in vert:
        file.write(' '.join(v)+'\n')

with open(output_edge, 'w') as file:
    for e in edge:
        file.write(str(e[0]) + ' ' + str(e[1]) + '\n')