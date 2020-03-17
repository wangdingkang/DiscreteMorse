import os
import csv
import math
from matplotlib import image as mpimg
import numpy as np
from os.path import join
from PIL import Image, ImageDraw
import PIL
import shutil
import sys
from scipy import misc

PIL.Image.MAX_IMAGE_PIXELS = None

IMAGE_DIR = sys.argv[1]
INPUT_FOLDER = sys.argv[2]
PATH_RADIUS = int(sys.argv[3])
output_dir = INPUT_FOLDER

# Files
VERT_FILENAME = join(INPUT_FOLDER, 'vert.txt')
EDGE_FILENAME = join(INPUT_FOLDER, 'no_dup_edge.txt')
PATH_FILENAME = join(INPUT_FOLDER, 'paths.txt')
GT_FILENAME = join(INPUT_FOLDER, 'diffusion_gt.txt')
DOMAIN_FILENAME = join(INPUT_FOLDER, 'diffusion_domain.txt')
output_filename = join(output_dir, 'output.txt')


def compute_abs_cos_angle(v1, v2):
    v1_array = np.asarray(v1)
    v2_array = np.asarray(v2)

    if np.linalg.norm(v1_array) == 0 or np.linalg.norm(v2_array) == 0:
        return 0

    v1_unit = v1_array / np.linalg.norm(v1_array)
    v2_unit = v2_array / np.linalg.norm(v2_array)
    angle = np.arccos(np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0))
    cos = math.cos(angle)
    return abs(cos)


def compute_tangents(p):
    estimated_tangents = []
    for i in range(len(p)):
        left = max(0, i - PATH_RADIUS)
        right = min(i + PATH_RADIUS, len(p) - 1)
        lv = p[left]
        rv = p[right]
        vector = (lv[0] - rv[0], lv[1] - rv[1], lv[2] - rv[2])
        estimated_tangents.append(vector)
    return estimated_tangents


verts = []
#sys.stdout.flush()
print('reading in verts...')
#sys.stdout.flush()
with open(VERT_FILENAME, 'r') as vert_file:
    reader = csv.reader(vert_file, delimiter=' ')
    for line in reader:
        verts.append((int(line[0]), int(line[1]), int(line[2])))
    vert_file.close()
#sys.stdout.flush()
print('VERTS', len(verts))
#sys.stdout.flush()


edges = []
print('reading in edges...')
with open(EDGE_FILENAME, 'r') as edge_file:
    reader = csv.reader(edge_file, delimiter=' ')
    for line in reader:
        v0 = int(line[0])
        v1 = int(line[1])
        if v0 < v1:
            edges.append((v0, v1, 0))
        else:
            edges.append((v1, v0, 1))
    edge_file.close()

intensity = {}
#sys.stdout.flush()
print('reading image...')
#sys.stdout.flush()

d_name = IMAGE_DIR
name_list = [name for name in os.listdir(d_name) if
             (os.path.isfile(d_name + '/' + name)) and (name != ".DS_Store")]
name_list.sort()

# print(misc.imread(os.path.join(d_name, name_list[0]), mode='F').shape)
nx, ny = mpimg.imread(os.path.join(d_name, name_list[0])).shape
nz = len(name_list)

im_cube = np.zeros([nx, ny, nz])
i = 0

for name in name_list:
    fileName = d_name + '/' + name
    # image = misc.imread(fileName, mode='F')
    image = mpimg.imread(fileName)
    im_cube[:, :, i] = image
    i = i + 1

# filtered_img = gaussian_filter(scaled_input_img, SIGMA)

for v in verts:
    x = v[0]
    y = v[1]
    z = v[2]
    f = im_cube[x, y, z]
    # print('f', f)
    intensity[v] = f

raw_paths = []
#sys.stdout.flush()
print('reading in paths...')
#sys.stdout.flush()
lines = [line.rstrip('\n').split(' ') for line in open(PATH_FILENAME)]

for i in range(len(lines)):
    lines[i] = lines[i][:len(lines[i]) - 1]

for line in lines:
    raw_paths.append([int(x) for x in line])
# paths = raw_paths

paths = []
#sys.stdout.flush()
print('computing valid paths...')
#sys.stdout.flush()
for p in raw_paths:
    if len(p) <= 1:
        print('path of len 1 or 0')
        sys.exit()
    paths.append(p)

#sys.stdout.flush()
print(len(paths), 'valid paths')
#sys.stdout.flush()


#sys.stdout.flush()
print('reading domain...')
#sys.stdout.flush()
domain = []
with open(DOMAIN_FILENAME, 'r') as domain_file:
    reader = csv.reader(domain_file, delimiter=' ')
    for line in reader:
        domain.append((int(line[0]), int(line[1]), int(line[2])))
    domain_file.close()
#sys.stdout.flush()
print('DOMAIN', len(domain))
#sys.stdout.flush()

#sys.stdout.flush()
print('reading vectors...')
#sys.stdout.flush()
vectors = []
with open(GT_FILENAME, 'r') as gt_file:
    reader = csv.reader(gt_file, delimiter=' ')
    for line in reader:
        vectors.append((float(line[0]), float(line[1]), float(line[2])))
    gt_file.close()

assert(len(domain) == len(vectors))

gt_dict = {}
#sys.stdout.flush()
print('building dict...')
#sys.stdout.flush()
for i in range(len(verts)):
    gt_dict[verts[i]] = vectors[i]

del domain
del vectors

#sys.stdout.flush()
print('ready to go!')
#sys.stdout.flush()
scores = []
lengths = []
degree_dict = {}
for i in range(len(verts)):
    degree_dict[i] = 0

e_info = {}
print(len(paths), 'paths')
for id, path in enumerate(paths):
    if id % 500 == 0:
        print(id, 'th path')
    v_path = [verts[i] for i in path]
    f_vals = [intensity[v] for v in v_path]
    # print(f_vals)
    tangents = compute_tangents(v_path)
    abs_cosines = []
    for i in range(len(v_path)):
        v = v_path[i]
        gt = gt_dict[v]
        tangent = tangents[i]
        abs_cosines.append(compute_abs_cos_angle(gt, tangent))

    for i in range(len(path)-1):
        vi = path[i]
        vip1 = path[i+1]
        if vi < vip1:
            v0 = vi
            f0 = f_vals[i]
            cos0 = abs_cosines[i]
            v1 = vip1
            f1 = f_vals[i+1]
            cos1 = abs_cosines[i+1]
        else:
            v0 = vip1
            f0 = f_vals[i+1]
            cos0 = abs_cosines[i+1]
            v1 = vi
            f1 = f_vals[i]
            cos1 = abs_cosines[i]

        edge_index = edges.index((v0, v1, 0))
        e_info[edge_index] = [f0, f1, cos0, cos1]

#sys.stdout.flush()
print('outputting...')
#sys.stdout.flush()

with open(output_filename, 'w') as output_filename:
    for i in range(len(edges)):
        info = e_info[i]
        if edges[i][2] == 1:
            print('fatal error')
            sys.exit()
        output_filename.write(str(edges[i][0]) + ' ' + str(edges[i][1]) + ' ' + str(info[0]) + ' ' + str(info[1]) + ' ' + str(info[2]) + ' ' + str(info[3]) + '\n')