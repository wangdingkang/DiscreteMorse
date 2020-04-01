import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('gt_swc', help=r"Path to the ground-truth swc.")
parser.add_argument('pre_swc', help=r"Path to the swc result.")
args = parser.parse_args()

input_gt_swc = args.gt_swc
input_pre_swc = args.pre_swc

# fix parameters for fMOST dataset.
MICRON_X, MICRON_Y, MICRON_Z = 0.35, 0.35, 1
radius = 4


if __name__ == '__main__':
    gt_point_list, good_gt, FN = [], [], []
    pre_point_list, TP, FP = [], [], []
    with open(input_gt_swc) as gt_file, open(input_pre_swc) as pre_file:
        for line in gt_file:
            if not line.startswith('#'):
                d = list(line.strip().split())
                gt_point_list.append((float(d[2]) * MICRON_X, float(d[3]) * MICRON_Y, float(d[4]) * MICRON_Z))
        for line in pre_file:
            if not line.startswith('#'):
                d = list(list(line.strip().split()))
                pre_point_list.append((float(d[2]) * MICRON_X, float(d[3]) * MICRON_Y, float(d[4]) * MICRON_Z))

    pre_tree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pre_point_list)
    gt_tree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(gt_point_list)

    distances, indices = pre_tree.kneighbors(gt_point_list)
    distances = [d[0] for d in distances]

    for i in range(len(gt_point_list)):
        if distances[i] <= radius:
            good_gt.append(gt_point_list[i])
        else:
            FN.append(gt_point_list[i])

    distances, indices = gt_tree.kneighbors(pre_point_list)
    distances = [d[0] for d in distances]
    for i in range(len(pre_point_list)):
        if distances[i] <= radius:
            TP.append(pre_point_list[i])
        else:
            FP.append(pre_point_list[i])
    precision, recall = len(TP) / (len(TP) + len(FP)), len(TP) / (len(TP) + len(FN))
    print('Precision, Recall, F1', precision, recall, 2 * precision * recall / (precision + recall))