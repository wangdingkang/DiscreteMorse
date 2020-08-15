import os
import sys
from collections import deque
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import numpy as np
from math import sqrt
import argparse

# We start to remove vertices from the leaves.
# Create two sets L and B stand for leave set and branching point set.
# For each vertex we compute a score, defined as:
# two directions, going to the root, and going to the leaves, Set a radius = r, score = avg(nodes on two directions within
# length r (r nodes))
# Notice that for a branching point (or close to branching point), there might be multiple paths going to leaves, take into
# account of all of them.

######## Global Parameters############

parser = argparse.ArgumentParser()
parser.add_argument('dir_name', help=r"Path to the directory of input files (image_stack, stree_e(v)).")
parser.add_argument('method', help='Method, ROOT or LEAF')
parser.add_argument('-n', dest='tree_num', type=int, default=0, help='Output from shortest path forest might has multiple trees, specify which tree you want to simplify (default is 0).')
parser.add_argument('-r', dest='resolution', type=float, nargs=3, default=[10,10,50], help='Resolution for the input images (default 10um, 10um, 50um).')
parser.add_argument('-score', dest='score', type=float, nargs=3, default=[10, 0.2, 50], help='[smoothing hop, simplification_threshold, score_assignment_cap] ([10,0.2,50]).')
parser.add_argument('-thickness', dest='thickness', type=float, nargs=2, default=[1,20], help='[thickness_constant, thickness_distance_cap], [1,20].')
parser.add_argument('-weight_cap', dest='weight_cap', type=float, default=300, help='distance cap for assigning summarization weight.')
args = parser.parse_args()


dirname = args.dir_name
tree_num = args.tree_num
METHOD = args.method
# Radius for smoothing process, and use threshold = FACTOR * avg_score to remove low-score nodes.
RADIUS, THRESHOLD_FACTOR, SCORE_DISTANCE_BOUND = args.score
MICRON_X, MICRON_Y, MICRON_Z = args.resolution
THICKNESS_FACTOR, THICKNESS_ASSIGNMENT_DISTANCE_UPPER_BOUND = args.thickness
WEIGHT_ASSIGNMENT_DISTANCE_UPPER_BOUND = args.weight_cap

# x y z
input_vert = os.path.join(dirname, 'stree_v{}.txt'.format(tree_num))
# u v f0 f1 c0 c1
input_edge = os.path.join(dirname, 'stree_e{}.txt'.format(tree_num))
# output simplified swc.
output_swc = os.path.join(dirname, 'stree_simp.swc')
# input Image stack to get voxel info.
image_stack_folder = os.path.join(dirname, 'image_stack')
# output voxel
output_voxel_path = os.path.join(dirname, 'simplifier_voxels.txt')
# output tree_nodes
output_tree_nodes_path = os.path.join(dirname, 'simplifier_tree_nodes.txt')

# threshold for removing low-density points, and upper bound for nn distance, pair with larger dist won't be considered.
# distance is in micron.
DENSITY_TRESHOLD = 0

# STP 15, 1.4, 100, 2, 'LEAF', c=1
# M91_1 5 1 5 2 leaf 1/3
# M91_2 10 1 5 2 leaf 1/3

# Class TreeNode. A node on the neuron tree.
# Two directional traversal availability.
# Score is used to decide burn the node or not.
class TreeNode:

    def __init__(self, id, x, y, z, dist):
        self.id, self.x, self.y, self.z = id, x, y, z
        self.children = [] # a list of TreeNodes
        self.father = None
        self.dist = dist # distance to root

        self.ancestor_cnt = 0
        self.path_intensity = 0

        self.sum_nn_intensity = 0   # T(v)
        self.avg_nn_intensity = 0   # I(v)
        self.thickness = 0
        self.summarization_weight = 0
        self.nn_cnt = 0             # C(v)
        self.smoothed_nn_cnt = 0    # newC(v)

        self.intensity = 0
        self.cosin = 0
        self.smoothed_intensity = 0
        self.smoothed_cosin = 0

        self.sum_nn_score = 0 # $newT(v)$
        self.avg_nn_score = 0 # $newI(v)$
        self.intensity_cos_score = 0


    def add_child(self, child):
        self.children.append(child)

    # Thickness for each tree_ndoe.
    def cal_thickness(self):
        self.thickness = sqrt(self.nn_cnt) * THICKNESS_FACTOR
        print(self.thickness)
        if self.thickness < 1:
            self.thickness = 1

    def cal_summarization_weights(self):
        self.summarization_weight = self.sum_nn_intensity

    def init_scores(self):
        self.sum_nn_intensity = 0  # T(v)
        self.avg_nn_intensity = 0  # I(v)
        self.nn_cnt = 0  # C(v)

    # which score for simp
    def get_score(self):
        return self.sum_nn_score
        # return self.path_intensity / self.ancestor_cnt

    # how to compute inten_cos score????
    def cal_intencos_scores(self):
        self.intensity_cos_score = self.smoothed_intensity * self.smoothed_cosin

    # output to simp_vert file, sum_nn_intensity used for selecting branches...
    def to_output_string(self):
        global MICRON_X, MICRON_Y, MICRON_Z
        return '{} {} {} {:2f} {:4f}\n'.format(self.x, self.y, self.z, self.get_score(), self.dist)

    # output for visualization nn or intensity-cos score for each tree node.
    def to_csv_string(self):
        # flip x & y
        return '{},{},{},{:2f},{:2f},{:2f}\n'.format(self.x, self.y, self.z, self.sum_nn_score, self.intensity_cos_score, self.path_intensity / self.ancestor_cnt)


# Used for cal score on the second direction, post-order updates.
def merge_subtree(subtree_head, subtrees2merge, radius):
    if not subtrees2merge:
        return Subtree2Leaves(1, subtree_head.intensity, subtree_head.cosin, subtree_head.sum_nn_intensity, subtree_head.avg_nn_intensity, subtree_head.nn_cnt, {subtree_head}, subtree_head, 1)
    else:
        # update leaves for all subtrees2merge
        new_leafset, tot_intensity, tot_cos, tot_snn, tot_ann, tot_cnn, n, depth = {}, 0, 0, 0, 0, 0, 0, 0
        for subtree in subtrees2merge:
            if subtree.depth >= radius:
                subtree.pop_leaves()
                new_leafset.update(subtree.leaves)
                tot_intensity += subtree.total_intensity
                tot_cos += subtree.total_cos
                tot_snn += subtree.total_snn
                tot_ann += subtree.total_ann
                tot_cnn += subtree.total_cnn
                n += subtree.n
                depth = max(depth, subtree.depth)
        tot_intensity += subtree_head.intensity
        tot_cos += subtree_head.cosin
        tot_snn += subtree_head.sum_nn_intensity
        tot_ann += subtree_head.avg_nn_intensity
        tot_cnn += subtree_head.nn_cnt
        return Subtree2Leaves(n + 1, tot_intensity, tot_cos, tot_snn, tot_ann, tot_cnn, new_leafset, subtree_head, depth + 1)


class Subtree2Leaves:

    def __init__(self):
        self.n = 0 # size of subtree
        self.total_intensity, self.total_cos, self.total_snn, self.total_ann, self.total_cnn = 0, 0, 0, 0, 0
        self.leaves = set()
        self.header = None
        self.depth = 0 # depth

    def __init__(self, n, intensity, cos, snn, ann, cnn, leaves, header, depth):
        self.n = n
        self.total_intensity = intensity
        self.total_cos = cos
        self.total_snn = snn
        self.total_ann = ann
        self.total_cnn = cnn
        self.leaves = leaves
        self.header = header
        self.depth = depth

    # pop leaves and update other attributes.
    def pop_leaves(self):
        intensity_loss = sum([node.intensity for node in self.leaves])
        cos_loss = sum([node.cosin for node in self.leaves])
        snn_loss = sum([node.sum_nn_intensity for node in self.leaves])
        ann_loss = sum([node.avg_nn_intensity for node in self.leaves])
        cnn_loss = sum([node.nn_cnt for node in self.leaves])
        self.total_intensity -= intensity_loss
        self.total_cos -= cos_loss
        self.total_snn -= snn_loss
        self.total_ann -= ann_loss
        self.total_cnn -= cnn_loss
        size_loss = len(self.leaves)
        self.n -= size_loss
        self.leaves = set([node.father for node in self.leaves])
        self.depth -= 1


class Subpath2Root:
    def __init__(self):
        self.queue2root = deque([])
        self.total_intensity, self.total_cos, self.total_snn, self.total_ann, self.total_cnn = 0, 0, 0, 0, 0

    def add(self, tree_node, radius):
        self.queue2root.append(tree_node)
        self.total_intensity += tree_node.intensity
        self.total_cos += tree_node.cosin
        self.total_snn += tree_node.sum_nn_intensity
        self.total_ann += tree_node.avg_nn_intensity
        self.total_cnn += tree_node.nn_cnt
        if len(self.queue2root) > radius:
            self.total_intensity -= self.queue2root[0].intensity
            self.total_cos -= self.queue2root[0].cosin
            self.total_snn -= self.queue2root[0].sum_nn_intensity
            self.total_ann -= self.queue2root[0].avg_nn_intensity
            self.total_cnn -= self.queue2root[0].nn_cnt
            return self.queue2root.popleft()
        return None

    def recover(self, removed_head):
        if removed_head is not None:
            self.queue2root.appendleft(removed_head)
            self.total_intensity += removed_head.intensity
            self.total_cos += removed_head.cosin
            self.total_snn += removed_head.sum_nn_intensity
            self.total_ann += removed_head.avg_nn_intensity
            self.total_cnn += removed_head.nn_cnt
        self.total_intensity -= self.queue2root[-1].intensity
        self.total_cos -= self.queue2root[-1].cosin
        self.total_snn -= self.queue2root[-1].sum_nn_intensity
        self.total_ann -= self.queue2root[-1].avg_nn_intensity
        self.total_cnn -= self.queue2root[-1].nn_cnt
        self.queue2root.pop()

    def get_all_metrics(self):
        return self.total_intensity, self.total_cos, self.total_snn, self.total_ann, self.total_cnn

    def get_size(self):
        return len(self.queue2root)


class DFS_Node:

    def __init__(self, _node):
        self.tree_node = _node
        self.child_id = 0
        self.downstream_subtrees = []
        self.removed_head = None

    def get_info(self):
        return self.tree_node, self.child_id, self.downstream_subtrees, self.removed_head


def update_score(root, up_deque, radius):
    # tree_node, child_id, downstream_subtrees, removed_head
    node_stack = [DFS_Node(root)]
    while len(node_stack) != 0:
        tree_node, child_id, downstream_subtrees, removed_head = node_stack[-1].get_info()
        ups_intensity, ups_cos, ups_snn, ups_ann, ups_cnn = up_deque.get_all_metrics()
        ups_size = up_deque.get_size()
        if removed_head:
            up_deque.recover(removed_head)
        if len(tree_node.children) > child_id:
            child = tree_node.children[child_id]
            node_stack[-1].child_id += 1
            node_stack[-1].removed_head = up_deque.add(tree_node, radius)
            node_stack.append(DFS_Node(child))
        else:
            merged_tree = merge_subtree(tree_node, downstream_subtrees, radius)
            dns_subattrs = [(subtr.total_intensity, subtr.total_cos, subtr.total_snn, subtr.total_ann, subtr.total_cnn, subtr.n) for subtr in downstream_subtrees]
            dns_intensity, dns_cos, dns_snn, dns_ann, dns_cnn, dns_size = 0, 0, 0, 0, 0, 0
            if dns_subattrs:
                dns_intensity, dns_cos, dns_snn, dns_ann, dns_cnn, dns_size = [sum(x) for x in zip(*dns_subattrs)]
            area_size = ups_size + dns_size + 1
            area_intensity = tree_node.intensity + ups_intensity + dns_intensity
            area_cos = tree_node.cosin + ups_cos + dns_cos
            area_snn = tree_node.sum_nn_intensity + ups_snn + dns_snn
            area_ann = tree_node.avg_nn_intensity + ups_ann + dns_ann
            area_cnn = tree_node.nn_cnt + ups_cnn + dns_cnn

            tree_node.smoothed_intensity = area_intensity / area_size
            tree_node.smoothed_cosin = area_cos / area_size
            tree_node.sum_nn_score = area_snn / area_size   #newT(v)
            tree_node.avg_nn_score = area_ann / area_size   #newI(v)
            tree_node.smoothed_nn_cnt = area_cnn / area_size #newC(v)

            tree_node.cal_intencos_scores()
            node_stack.pop()
            if len(node_stack) != 0:
                node_stack[-1].downstream_subtrees.append(merged_tree)


# Maintain two sets L and B for leave burning process.
def leaf_burner(L, score_threshold):
    B = {}
    Removed = set()
    while L:
        C = set()
        for leaf in L:
            if leaf.get_score() < score_threshold:
                Removed.add(leaf.id)
                father = leaf.father
                cnt_children = len(father.children)
                if cnt_children == 1:
                    C.add(father)
                else:
                    if father in B.keys():
                        B[father] = B[father] - 1
                        if B[father] == 0:
                            C.add(father)
                    else:
                        B[father] = cnt_children - 1
        L = C
    return Removed


def root_expander(R, score_threshold):
    Kept = set()
    while len(R) != 0:
        C = set()
        for node in R:
            if node.get_score() >= score_threshold:
                Kept.add(node.id)
                for child in node.children:
                    C.add(child)
        R = C
    return Kept


def read_verts_edges(input_vpath, input_epath):
    tree_nodes = []
    with open(input_vpath, 'r') as file:
        cnt = 0
        for line in file:
            data = line.strip().split()[:4]
            x, y, z = list(map(int, data[:3]))
            dist = float(data[3])
            tree_nodes.append(TreeNode(cnt, x, y, z, dist))
            cnt += 1

    edge_list = [[] for i in range(cnt)]
    with open(input_epath, 'r') as file:
        for line in file:
            u, v, f0, f1, c0, c1 = line.strip().split()
            u, v = int(u), int(v)
            f0, f1, c0, c1 = float(f0), float(f1), float(c0), float(c1)
            edge_list[u].append(v)
            edge_list[v].append(u)
            tree_nodes[u].intensity, tree_nodes[u].cosin = f0, c0
            tree_nodes[v].intensity, tree_nodes[v].cosin = f1, c1
    return tree_nodes, edge_list


def update_connections(start_id, tree_nodes, edge_list, visited):
    node_stack = [start_id]
    while len(node_stack) != 0:
        temp_id = node_stack.pop()
        visited[temp_id] = True
        for e in edge_list[temp_id]:
            if not visited[e]:
                tree_nodes[temp_id].add_child(tree_nodes[e])
                tree_nodes[e].father = tree_nodes[temp_id]
                node_stack.append(e)


CNT = 0
def swc_dfs_tree(remaining_id, nodes, res, tid=0, pcnt=-1 ):
    global CNT
    # new_id child_pivot old_id
    res.append('#id,type,x,y,z,thickness,parent_id,summary_weight\n')
    stack = [[tid, 0, 0]]
    res.append(
        '{} {} {} {} {} {:2f} {} {}\n'.format(0, 1, nodes[0].y, nodes[0].x, nodes[0].z, nodes[0].thickness, -1, nodes[0].summarization_weight))

    while len(stack) > 0:
        temp = stack[-1]
        tnode = nodes[temp[2]]
        if temp[1] < len(tnode.children):
            next_node = tnode.children[temp[1]]
            temp[1] += 1
            if next_node.id in remaining_id:
                CNT += 1
                res.append('{} {} {} {} {} {:2f} {} {}\n'.format(CNT, 1, next_node.y, next_node.x, next_node.z, next_node.thickness, temp[0], next_node.summarization_weight))
                stack.append([CNT, 0, next_node.id])
        else:
            stack.pop()




def output_simptree(output_swc, origin_tree_nodes, edge_list, removed_node_ids):
    n = len(origin_tree_nodes)
    id_mapping, cnt, remaining_id = {}, 0, list(set([j for j in range(n)]).difference(removed_node_ids))
    global CNT
    swc_res, CNT = [], 0
    swc_dfs_tree(remaining_id, origin_tree_nodes, swc_res)
    with open(output_swc, 'w') as file:
        for line in swc_res:
            file.write(line)


# voxel = list of (x, y, z, intensity)
# tree_nodes list of (x, y, z)
def update_nearest_neighbor_intensity(tree_nodes, voxel_list, radius_limit):
    global MICRON_X, MICRON_Y, MICRON_Z
    tree_nodes_coords = [[node.x * MICRON_X, node.y * MICRON_Y, node.z * MICRON_Z] for node in tree_nodes]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(np.array(tree_nodes_coords))
    voxel_coords = np.array([[v[0] * MICRON_X, v[1] * MICRON_Y, v[2] * MICRON_Z] for v in voxel_list])
    distances, indices = nbrs.kneighbors(voxel_coords)
    print('Computing NN for each voxel')
    total_weights, assigned_weights = 0, 0
    for i in range(len(voxel_list)):
        index, dist, weight = indices[i][0], distances[i][0], voxel_list[i][3]
        total_weights += weight
        if dist <= radius_limit:
            assigned_weights += weight
            tree_nodes[index].sum_nn_intensity += weight
            tree_nodes[index].nn_cnt += 1
    for tree_node in tree_nodes:
        if tree_node.nn_cnt != 0:
            tree_node.avg_nn_intensity = tree_node.sum_nn_intensity / tree_node.nn_cnt
    print('weights assigned', total_weights, assigned_weights, 'percentage:', assigned_weights / total_weights)


def update_sum_nn_along_path(start_id, tree_nodes):
    node_stack = [tree_nodes[start_id]]
    while len(node_stack) != 0:
        temp_node = node_stack.pop()
        temp_node.ancestor_cnt = 1
        temp_node.path_intensity = temp_node.sum_nn_intensity
        if temp_node.father != None:
            temp_node.ancestor_cnt += temp_node.father.ancestor_cnt
            temp_node.path_intensity += temp_node.father.path_intensity
        for next in tree_nodes[temp_node.id].children:
            node_stack.append(next)


def read_voxel_list(img_stack_folder, density_threshold):
    image_paths = sorted(os.listdir(img_stack_folder))
    z_range = range(1, len(image_paths) + 1)
    image_points, first_time = [], True
    print('Reading...')
    for img_name, z in zip(image_paths, z_range):
        img_path = os.path.join(img_stack_folder, img_name)
        img = Image.open(img_path)
        img_arr = np.array(img)
        points = np.nonzero(img_arr > density_threshold)
        z_axes = np.expand_dims(z * np.ones(len(points[0]), dtype=np.int64), axis=1)
        intensity = np.expand_dims(img_arr[points], axis = 1)
        points = np.transpose(points)
        points = np.concatenate((points, z_axes, intensity),  axis=1)
        image_points.extend(points.tolist())
    print('In total,', len(image_points), 'voxels')
    return image_points


def output_voxel_list(voxels, output_path):
    with open(output_path, 'w') as file:
        for v in voxels:
            file.write(f"{v[0]} {v[1]} {v[2]} {v[3]}" + "\n")

def output_tree_nodes(tree_nodes, output_path):
    with open(output_path, 'w') as file:
        for node in tree_nodes:
            file.write(node.to_output_string())

if __name__ == '__main__':

    voxel_list = read_voxel_list(image_stack_folder, DENSITY_TRESHOLD)
    tree_nodes, edge_list = read_verts_edges(input_vert, input_edge)
    output_voxel_list(voxel_list, output_voxel_path)
    output_tree_nodes(tree_nodes, output_tree_nodes_path)    
    root, root_id = tree_nodes[0], 0

    print('Updating intensity and connection')
    update_nearest_neighbor_intensity(tree_nodes, voxel_list, SCORE_DISTANCE_BOUND)

    visited = [False for i in range(len(tree_nodes))]
    update_connections(root_id, tree_nodes, edge_list, visited)

    print('Computing smoothed scores for each tree node...')
    assert (RADIUS > 1)
    update_score(root, Subpath2Root(), radius=RADIUS)
    print('Update sum nn along path...')
    update_sum_nn_along_path(root_id, tree_nodes)

    leaves = [node for node in tree_nodes if len(node.children) == 0]
    scores = [node.get_score() for node in tree_nodes]
    scores.sort()
    max_score = max(scores)
    median_score = scores[len(scores) // 2]
    avg_score = sum(scores) / len(tree_nodes)

    score_threhold = avg_score * THRESHOLD_FACTOR
    print('Average Score', avg_score, 'Max Score', max_score, 'Median Score', median_score)
    removed_id = None
    if METHOD == 'LEAF':
        print('Leaf Burner...')
        removed_id = leaf_burner(leaves, score_threhold)
    else:
        print('Root Expander...')
        removed_id = set([i for i in range(len(tree_nodes))]).difference(root_expander([tree_nodes[0]], score_threhold))

    simplified_treenodes = [node for node in tree_nodes if node.id not in removed_id]
    for node in simplified_treenodes:
        node.init_scores()
    update_nearest_neighbor_intensity(simplified_treenodes, voxel_list, THICKNESS_ASSIGNMENT_DISTANCE_UPPER_BOUND)
    print('Update thickness...')
    for node in simplified_treenodes:
        node.cal_thickness()
    for node in simplified_treenodes:
        node.init_scores()
    update_nearest_neighbor_intensity(simplified_treenodes, voxel_list, WEIGHT_ASSIGNMENT_DISTANCE_UPPER_BOUND)
    for node in simplified_treenodes:
        node.cal_summarization_weights()
    print('Had', len(tree_nodes), ', Removed', len(removed_id), 'Writing')
    output_simptree(output_swc, tree_nodes, edge_list, removed_id)