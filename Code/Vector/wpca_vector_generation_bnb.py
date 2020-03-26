import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import csv
import sys
import os, os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from wpca import WPCA
import matplotlib.image as mpimg

import PIL
PIL.Image.MAX_IMAGE_PIXELS = None

# HALF_CUBE_SIDE = 3
HALF_CUBE_SIDE = 5

'''
X_MIN = 475
X_MAX = 600
Y_MIN = 975
Y_MAX = 1075
MIN_Z = 0
MAX_Z = 100
'''

'''
X_MIN = 950
X_MAX = 1200
Y_MIN = 1950
Y_MAX = 2150
MIN_Z = 0
MAX_Z = 100
'''

'''
X_MIN = 5900
X_MAX = 6099
Y_MIN = 20300
Y_MAX = 20499
MIN_Z = 3401
MAX_Z = 3600
'''

# PLOT_CENTERS_FILENAME = '../results/1605_subset/vert.txt'
out_dir = sys.argv[1]
PLOT_CENTERS_FILENAME = out_dir + 'vert.txt'

# PLOT_CENTERS = [(540,1013,60),(558,1011,66),(528,995,54),(558,1004,61),(589,1008,74),(581,1027,68),(577,989,74),(514,1010,57),(485,1006,49),(489,992,47)]
# PLOT_CENTERS = [(540,1013,60)]

PC_MULT_FACTOR = 10


output_centers_filename = out_dir + "centers.txt"
output_verts_filename = out_dir + "vec_vert.txt"
output_edges_filename = out_dir + "vec_edge.txt"
output_pc_filename = out_dir + "pc.txt"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

d_name = sys.argv[2]
name_list = [name for name in os.listdir(d_name) if
             (os.path.isfile(d_name + '/' + name)) and (name != ".DS_Store")]
name_list.sort()
nFile = len(name_list)
nz = nFile
nx, ny = mpimg.imread(d_name + '/' + name_list[0]).shape

sys.stdout.flush()
print('reading in plot centers...')
sys.stdout.flush()
plot_centers = set()
with open(PLOT_CENTERS_FILENAME, 'r') as centers_file:
    reader = csv.reader(centers_file, delimiter=' ')
    for line in reader:
        pt = (int(line[0]), int(line[1]), int(line[2]))
        plot_centers.add(pt)
    
plot_centers = list(plot_centers)
print('centers read:', len(plot_centers))
sys.stdout.flush()

index = 0

with open(output_verts_filename, 'w') as verts_file:
    with open(output_edges_filename, 'w') as edge_file:
        with open(output_pc_filename, 'w') as pc_file:
            with open(output_centers_filename, 'w') as center_file:
                im_cube = np.zeros([nx, ny, nz])
                #im_cube = np.zeros([201, 201, 201])
                i = 0
                sys.stdout.flush()
                print('loading images...')
                sys.stdout.flush()
                for name in name_list:
                    print(name)
                    fileName = d_name + '/' + name
                    image = mpimg.imread(fileName)
                    im_cube[:, :, i] = image
                    i = i + 1
                sys.stdout.flush()
                print('images loaded')
                sys.stdout.flush()

                sys.stdout.flush()
                print('building verts...')
                sys.stdout.flush()

                for pt in plot_centers:
                    x = pt[0]
                    y = pt[1]
                    z = pt[2]

                    min_x = max(0, x - HALF_CUBE_SIDE)
                    max_x = min(nx-1, x + HALF_CUBE_SIDE)

                    min_y = max(0, y - HALF_CUBE_SIDE)
                    max_y = min(ny-1, y + HALF_CUBE_SIDE)

                    min_z = max(0, z - HALF_CUBE_SIDE)
                    max_z = min(nz-1, z + HALF_CUBE_SIDE)

                    sys.stdout.flush()
                    print(index+1,'/',len(plot_centers))
                    sys.stdout.flush()

                    nbhd = []
                    for i in range(min_x, max_x + 1):
                        for j in range(min_y, max_y + 1):
                            for k in range(min_z, max_z + 1):
                                nbhd.append([i,j,k,im_cube[i,j,k]])

                    nbhd_pts = [[n_pt[0], n_pt[1], n_pt[2]] for n_pt in nbhd]
                    coords = np.asarray(nbhd_pts)
                    vals = [n_pt[3] for n_pt in nbhd]
                    weights = []
                    for val in vals:
                        weights.append([val, val, val])
                    weights = np.asarray(weights)
                    pca = WPCA(n_components=1)
                    pca.fit(coords, weights=weights)
                    component = pca.components_[0]
                    sys.stdout.flush()
                    print('principle component:', component)
                    sys.stdout.flush()

                    edge_file.write(str(2 * index) + ' ' + str(2 * index + 1) + '\n')
                    verts_file.write(str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]) + '\n')
                    pc_file.write(str(component[0]) + ' ' + str(component[1]) + ' ' + str(component[2]) + '\n')
                    center_file.write(str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]) + '\n')

                    scaled_component = [PC_MULT_FACTOR * j for j in component]
                    pt2 = [pt[j] + scaled_component[j] for j in range(len(pt))]
                    verts_file.write(str(pt2[0]) + ' ' + str(pt2[1]) + ' ' + str(pt2[2]) + '\n')
                    index += 1
                center_file.close()
            pc_file.close()
        edge_file.close()
    verts_file.close()
