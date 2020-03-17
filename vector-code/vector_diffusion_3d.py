import matplotlib
matplotlib.use('agg')
import matplotlib.image as mpimg

import csv
from math import exp
import sys
import os
import numpy as np
import scipy.spatial as spatial

SIGMA = 2

PC_MULT_FACTOR = 10
d_name = sys.argv[2]
name_list = [name for name in os.listdir(d_name) if
             (os.path.isfile(d_name + '/' + name)) and (name != ".DS_Store")]
name_list.sort()

nx, ny = mpimg.imread(d_name + '/' + name_list[0]).shape
nz = len(name_list)

#im_cube = np.zeros([456, 343, 270])
#im_cube = np.zeros([201, 201, 201])
im_cube = np.zeros([nx, ny, nz])
i = 0

for name in name_list:
    fileName = d_name + '/' + name
    image = mpimg.imread(fileName)
    im_cube[:, :, i] = image
    i = i + 1


def distance(pt1, pt2):
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2 + (pt1[2] - pt2[2]) ** 2) ** .5


def compute_vector_diffuison(pt, kd_tree, dict):
    x, y, z = pt
    threshold = 3 * SIGMA
    vector = [0, 0, 0]
    close_points = kd_tree.data[kd_tree.query_ball_point(pt, threshold)]
    val = im_cube[x, y, z]
    for close_point in close_points:
        dist = distance(pt, close_point)
        factor = val * exp(-dist/SIGMA)
        value = dict[tuple(close_point)]
        vector[0] += factor * value[0]
        vector[1] += factor * value[1]
        vector[2] += factor * value[2]
    magnitude = (vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2) ** .5
    if magnitude == 0:
        normalized = (0, 0, 0)
    else:
        normalized = (vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude)
    return normalized


directory = sys.argv[1]
center_filename = directory + "centers.txt"
pc_filename = directory + "pc.txt"
vert_filename = directory + "diffusion_vert.txt"
edge_filename = directory + "diffusion_edge.txt"
vector_filename = directory + "diffusion_gt.txt"
domain_filename = directory + "diffusion_domain.txt"

print('loading...')
centers = []
wpcs = []
center_wpca_dict = {}
with open(center_filename, 'r') as center_file:
    reader = csv.reader(center_file, delimiter=' ')
    for x in reader:
        centers.append((int(x[0]), int(x[1]), int(x[2])))

with open(pc_filename, 'r') as pc_file:
    reader = csv.reader(pc_file, delimiter=' ')
    for x in reader:
        x_axis_component = float(x[0])
        if x_axis_component >= 0:
            wpcs.append((float(x[0]), float(x[1]), float(x[2])))
        else:
            wpcs.append((-float(x[0]), -float(x[1]), -float(x[2])))
    
print('loaded')

assert(len(centers) == len(wpcs))

for i in range(len(centers)):
    center_wpca_dict[centers[i]] = wpcs[i]


diffusion_dict = {}
print('computing')
ind = 1

center_np = np.array(centers)
center_tree = spatial.cKDTree(center_np)
print(len(centers), 'points')
for pt in centers:
    if ind % 1000 == 0:
        print('Working on point', ind)
    x, y, z = pt
    diffusion_vector = compute_vector_diffuison(pt, center_tree, center_wpca_dict)
    diffusion_dict[(x, y, z)] = diffusion_vector
    ind += 1
print('computed')
sys.stdout.flush()
index = 0
with open(vert_filename, 'w') as vert_file:
    with open(edge_filename, 'w') as edge_file:
        with open(domain_filename, 'w') as domain_file:
            with open(vector_filename, 'w') as vector_file:
                for key in diffusion_dict.keys():
                    vert_file.write(str(key[0]) + ' ' + str(key[1]) + ' ' + str(key[2]) + '\n')
                    val = diffusion_dict[key]
                    scaled_val = [PC_MULT_FACTOR * j for j in val]
                    pt2 = [key[j] + scaled_val[j] for j in range(len(key))]
                    vert_file.write(str(pt2[0]) + ' ' + str(pt2[1]) + ' ' + str(pt2[2]) + '\n')
                    edge_file.write(str(2 * index) + ' ' + str(2 * index + 1) + '\n')
                    domain_file.write(str(key[0]) + ' ' + str(key[1]) + ' ' + str(key[2]) + '\n')
                    vector_file.write(str(val[0]) + ' ' + str(val[1]) + ' ' + str(val[2]) + '\n')
                    index += 1