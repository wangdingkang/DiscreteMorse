import sys
import csv
import os

dir_name = sys.argv[1]
input_filename = os.path.join(dir_name, 'edge.txt')
output_filename = os.path.join(dir_name, 'no_dup_edge.txt')
print(input_filename, output_filename)

with open(input_filename, 'r') as input_file:
    arr = []
    reader = csv.reader(input_file, delimiter=' ')
    for row in reader:
        v0 = int(row[0])
        v1 = int(row[1])
        if v0 < v1:
            vmin = v0
            vmax = v1
        else:
            vmin = v1
            vmax = v0
        arr.append((vmin, vmax))
edges = list(set(arr))
edges.sort()   

with open(output_filename, 'w') as output_file:
    for e in edges:
        output_file.write(str(e[0]) + ' ' + str(e[1]) + '\n')
   
