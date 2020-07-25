import os
import sys
import numpy
import argparse
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors


parser = argparse.ArgumentParser()
parser.add_argument('dir_name', help=r"Path to the directory of images, the images should be stored under 'image_stack' subfolder.")
parser.add_argument('-t', dest='distance_threshold', help='Max distance to non-zero voxels. Any faraway edges will be removed (default 5).', type=float, default=5)
args = parser.parse_args()


dir_name = args.dir_name
threshold = args.distance_threshold
image_stack = os.path.join(dir_name, 'image_stack')
input_vfile = os.path.join(dir_name, 'vert.txt')
input_efile, output_file = os.path.join(dir_name, 'output.txt'), os.path.join(dir_name, 'edge.txt')

edges_left = []
cnt = 0


def read_voxel_list(img_stack_folder):
    image_paths = sorted(os.listdir(img_stack_folder))
    z_range = range(1, len(image_paths) + 1)
    image_points, first_time = [], True
    for img_name, z in zip(image_paths, z_range):
        print(img_name)
        img_path = os.path.join(img_stack_folder, img_name)
        img = Image.open(img_path)
        img_arr = np.array(img)
        points = np.nonzero(img_arr > 0)
        z_axes = np.expand_dims(z * np.ones(len(points[0]), dtype=np.int64), axis=1)
        intensity = np.expand_dims(img_arr[points], axis = 1)
        points = np.transpose(points)
        points = np.concatenate((points, z_axes),  axis=1)
        image_points.extend(points.tolist())
    print('In total,', len(image_points), 'voxels')
    return image_points


if __name__ == '__main__':
    voxels = read_voxel_list(image_stack)
    mx = 0
    for v in voxels:
        mx = max(mx, v[0])
    tree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(np.array(voxels))
    verts = []
    with open(input_vfile) as file:
        for line in file:
            data = line.strip().split()
            verts.append(list(map(int, data[:3])))
    with open(input_efile, 'r') as file:
        for line in file:
            cnt += 1
            data = line.strip().split()
            f1, f2 = float(data[2]), float(data[3])
            u, v = int(data[0]), int(data[1])
            # not on z bottom
            if verts[u][2] != 0 and verts[v][2] != 0:
                distances, indices = tree.kneighbors(np.array([verts[u]]))
                if distances[0][0] < threshold:
                    edges_left.append(line)
    print(cnt, 'edges.')
    with open(output_file, 'w') as file:
        print(len(edges_left), 'edges left.')
        for d in edges_left:
            file.write(d)





