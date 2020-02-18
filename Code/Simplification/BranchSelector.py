import os
import sys
import itertools
from math import sqrt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dir_name', help=r"Path to the directory of input files (image_stack, stree_e(v)).")
parser.add_argument('-n', dest='branch_num', type=int, default=20, help='Number of top branches.')

args = parser.parse_args()

# Working Directory
dir_name = args.dir_name
bcnt = args.branch_num

# path to simplified vert file & edge file
# input_vpath, input_epath = os.path.join(dir_name, 'stree_vsimp0.txt'), os.path.join(dir_name, 'stree_esimp0.txt')
input_swcpath = os.path.join(dir_name, 'stree_simp.swc')
output_swc = os.path.join(dir_name, 'stree_branch' + str(bcnt) + '.swc')


class TreeNode:

    def __init__(self, id, x, y, z, score, dist):
        self.id, self.x, self.y, self.z = id, x, y, z
        self.children = [] # a list of TreeNodes
        self.father = None
        self.score = score
        self.total_score = score
        self.total_dist = dist

    def add_child(self, child):
        self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0

    def is_branching(self):
        return len(self.children) > 1

    def get_score(self):
        return self.total_dist

    def update_score(self, value):
        self.total_dist += value

    def set_score(self, value):
        self.total_dist = value


def read_verts_edges(input_swc):
    tree_nodes, edges = [], []
    with open(input_swc) as file:
        for line in file:
            if not line.startswith('#'):
                data = line.strip().split()
                x, y, z = map(int, data[2:5])
                id, pid = int(data[0]), int(data[-2])
                tree_nodes.append(TreeNode(id, x, y, z, 0, 0))
                if pid != -1:
                    px, py, pz = tree_nodes[pid].x, tree_nodes[pid].y, tree_nodes[pid].z
                    tree_nodes[id].total_dist = sqrt((px-x)**2+(py-y)**2+(pz-z)**2) + tree_nodes[pid].total_dist
                    edges.append([id, pid])

    edge_list = [[] for i in range(len(tree_nodes))]
    for e in edges:
        u, v = e
        edge_list[u].append(v)
        edge_list[v].append(u)

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
                tree_nodes[e].total_score += tree_nodes[temp_id].total_score
                node_stack.append(e)


def down_update(start_id, tree_nodes, visited):
    node_stack = [start_id]
    amount = tree_nodes[start_id].get_score()
    while len(node_stack) != 0:
        temp_id = node_stack.pop()
        for child in tree_nodes[temp_id].children:
            if not visited[child.id]:
                tree_nodes[child.id].update_score(-amount)
                node_stack.append(child.id)


def select_branches(tree_nodes, cnt_branches = -1):
    leaves = set([i for i, _ in enumerate(tree_nodes) if tree_nodes[i].is_leaf()])
    selected_branches = []
    print(len(leaves))
    visited = [False for i in range(len(tree_nodes))]
    if cnt_branches == -1:
        cnt_branches = len(leaves)
    for i in range(cnt_branches):
        print('Selecting', i, 'th branch..')
        temp_score, temp_id, temp_branch = -1, -1, []
        for leaf in leaves:
            if tree_nodes[leaf].get_score() > temp_score:
                temp_score, temp_id = tree_nodes[leaf].get_score(), tree_nodes[leaf].id
        print('Total for this branch,', temp_score)
        leaves.remove(temp_id)
        # update
        while not visited[temp_id]:
            visited[temp_id] = True
            if tree_nodes[temp_id].is_branching():
                down_update(temp_id, tree_nodes, visited)
            tree_nodes[temp_id].set_score(0)
            if temp_id != 0:
                temp_branch.append([tree_nodes[temp_id].father.id, temp_id])
                temp_id = tree_nodes[temp_id].father.id
            else:
                break
        temp_branch.reverse()
        selected_branches.append(temp_branch)
    return selected_branches


def output_branches_to_swc(selected_branches, whole_swc_file, output_path):
    selected_ids = set([e for item in selected_branches for t in item for e in t])
    data = []
    with open(whole_swc_file) as file:
        for line in file:
            data.append(line)
    data = [data[i] for i in selected_ids]
    with open(output_path, 'w') as file:
        file.write(''.join(data))


def output_branches_by_group(selected_branches, output_prefix, stride_size = 5):
    temp_edges, cnt = [], 0
    for i in range(len(selected_branches)):
        if i != 0 and i % stride_size == 0:
            with open(output_prefix + str(cnt) + '.txt', 'w') as file:
                for e in temp_edges:
                    file.write('{} {}\n'.format(e[0], e[1]))
            temp_edges = []
            cnt += 1
        temp_edges.extend(selected_branches[i])
    if len(temp_edges) != 0:
        with open(output_prefix + str(cnt) + '.txt', 'w') as file:
            for e in temp_edges:
                file.write('{} {}\n'.format(e[0], e[1]))



if __name__ == '__main__':
    tree_nodes, edge_list = read_verts_edges(input_swcpath)
    update_connections(0, tree_nodes, edge_list, [False for i in range(len(tree_nodes))])
    selected_branches = select_branches(tree_nodes, bcnt)
    output_branches_to_swc(selected_branches, input_swcpath, output_swc)