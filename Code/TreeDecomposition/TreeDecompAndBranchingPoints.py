import numpy as np
import cvxpy as cp
import math
import os
from PIL import Image
import pdb
import cvxopt
from util.swc_branching_points import SWCBranchingPoint, TreeNode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb

class Matching:
    def __init__(self, dm_point_id):
        self.source_node_id = dm_point_id
        self.type = 3  # no matching
        self.branch1_matching, self.branch2_matching = {}, {}
        self.split_matching = {}
        self.y_shape = None

    def get_info_string(self):
        return "It is a type-{type} matching for point {point_id}.\n".format(type=self.type,
                                                                             point_id=self.source_node_id)

    def get_y_shape(self):
        return self.y_shape

    def get_matching_branchs(self):
        if self.type == 1:
            return self.branch1_matching, self.branch2_matching
        if self.type == 2:
            return self.branch1_matching, self.branch2_matching
        return None, None

    def get_splitting_match(self):
        return self.split_matching


def construct_and_solve(mouselight_thicken_mats, target_mat):
    # pdb.set_trace()
    target_node_set = set([tuple(x) for x in (np.argwhere(target_mat > 0.1).tolist())])
    mouselight_node_sets = []
    for mouselight_mat in mouselight_thicken_mats:
        mouselight_node_sets.append(set([tuple(x) for x in (np.argwhere(mouselight_mat > 0.1).tolist())]))
    all_nodes = set()
    all_nodes.update(target_node_set)
    for s in mouselight_node_sets:
        all_nodes.update(s)
    node_set_flatten_mapping = {node: i for i, node in enumerate(all_nodes)}
    m = len(all_nodes)
    n = len(mouselight_node_sets)
    print(m, n)
    target_vec = np.zeros(m, dtype=np.float)
    target_indices = [node_set_flatten_mapping[node] for node in target_node_set]
    target_vec[target_indices] = 1.0

    tree_mat = np.zeros((m, n), dtype=np.float)
    for i, neuron_node_set in enumerate(mouselight_node_sets):
        this_neuron_vec = [node_set_flatten_mapping[node] for node in neuron_node_set]
        tree_mat[this_neuron_vec, i] = 1.0

    x, vlambda = cp.Variable(n), 0
    # construct Convex OPT problem.
    objective = cp.Minimize(cp.sum_squares(tree_mat @ x - target_vec) + vlambda * cp.norm(x, 1))
    constraints = [0 <= x, x <= 1]
    prob = cp.Problem(objective, constraints)
    print('Start solving...')
    prob.solve(solver=cp.SCS,  verbose=True, max_iters=10000)
    print('Solver status: {}'.format(prob.status))

    print("optimal objective value: {}".format(objective.value))
    print("A solution x is")
    print(x.value)
    return x.value


def find_splitting_point_type(dm_SWC, splitting_points, selected_mouselight_SWCs):
    matchings = []
    for splitting_point in splitting_points:
        point_id = splitting_point.id
        matching = Matching(point_id)
        y_shape = dm_SWC.get_split_branches(point_id)
        matching.y_shape = y_shape
        branch1, branch2 = y_shape['branch1'], y_shape['branch2']
        splitting_point = y_shape['splitting_point']
        for id, mouselight_swc in enumerate(selected_mouselight_SWCs):
            # the two branches is covered by a single mouselight neuron.
            is_matching_left, left_node_list = mouselight_swc.overlap_with_branch(branch1)
            is_matching_right, right_node_list = mouselight_swc.overlap_with_branch(branch2)
            is_matching_splitting, matched_splitting = mouselight_swc.matching_splitting_point(splitting_point)
            if is_matching_left and is_matching_right and is_matching_splitting:
                matching.type = 1
                matching.split_matching = {id: matched_splitting}
                matching.branch1_matching, matching.branch2_matching = {id: left_node_list}, {id: right_node_list}
                break
            if is_matching_left:
                matching.type = 2
                matching.branch1_matching[id] = left_node_list
            if is_matching_right:
                matching.type = 2
                matching.branch2_matching[id] = right_node_list
        matchings.append(matching)
    return matchings


def find_splitting_point_type_v1(dm_SWC, splitting_points, selected_mouselight_SWCs):
    matchings = []
    # splitting_points = [splitting_points[6]]
    for splitting_point in splitting_points:
        point_id = splitting_point.id
        matching = Matching(point_id)
        y_shape = dm_SWC.get_split_branches(point_id, hop_limit=60)
        matching.y_shape = y_shape
        branch1, branch2 = y_shape['branch1'], y_shape['branch2']
        splitting_point = y_shape['splitting_point']
        for id, mouselight_swc in enumerate(selected_mouselight_SWCs):
            close_splitting_points = mouselight_swc.get_matched_splitting_point(splitting_point)
            close_points = mouselight_swc.get_matched_point(splitting_point)
            for close_splitting_point in close_splitting_points:
                mouselight_y_shape = mouselight_swc.get_split_branches(close_splitting_point.id, hop_limit=30)
                mouselight_branch1, mouselight_branch2 = mouselight_y_shape['branch1'], mouselight_y_shape['branch2']
                b1_mb1, b2_mb2= branch2branch_matching(branch1, mouselight_branch1), branch2branch_matching(branch2, mouselight_branch2)
                b1_mb2, b2_mb1 = branch2branch_matching(branch1, mouselight_branch2), branch2branch_matching(branch2, mouselight_branch1)
                # pdb.set_trace()
                if b1_mb1 and b2_mb2:
                    matching.type = 1
                    matching.split_matching = {id: close_splitting_point.get_coordinate()}
                    matching.branch1_matching[id] = mouselight_branch1
                    matching.branch2_matching[id] = mouselight_branch2
                    break
                if b1_mb2 and b2_mb1:
                    matching.type = 1
                    matching.split_matching = {id: close_splitting_point.get_coordinate()}
                    matching.branch1_matching[id] = mouselight_branch2
                    matching.branch2_matching[id] = mouselight_branch1
                    break
                if b1_mb1 or b1_mb2:
                    matching.type = 2
                    if b1_mb1:
                        matching.branch1_matching[id] = mouselight_branch1
                    if b1_mb2:
                        matching.branch1_matching[id] = mouselight_branch2

                if b2_mb1 or b2_mb2:
                    matching.type = 2
                    if b2_mb1:
                        matching.branch2_matching[id] = mouselight_branch1
                    if b2_mb2:
                        matching.branch2_matching[id] = mouselight_branch2
            if matching.type == 1:
                break
            for close_point in close_points:
                mouselight_branch = mouselight_swc.get_single_branch(close_point.id, hop_limit=30)
                b1_mb = branch2branch_matching(branch1, mouselight_branch)
                b2_mb = branch2branch_matching(branch2, mouselight_branch)
                if b1_mb:
                    matching.type = 2
                    matching.branch1_matching[id] = mouselight_branch
                    break
                if b2_mb:
                    matching.type = 2
                    matching.branch2_matching[id] = mouselight_branch
                    break
        matchings.append(matching)
    return matchings

def get_distance(point_a, point_b):
    dx, dy, dz = point_a[0] - point_b[0], point_a[1] - point_b[1], point_a[2] - point_b[2]
    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

def branch2branch_matching(source_branch, swc_branch, dist_bound = 5):
    total_cnt = len(source_branch)
    right_cnt = 0
    for point in source_branch:
        for target_point in swc_branch:
            if get_distance(point, target_point) < dist_bound:
                right_cnt += 1
                break
    return right_cnt / total_cnt > 0.5

def visualize_matchings(matchings, swc_file_names, source_SWC):
    for id, matching in enumerate(matchings):
        if matching.type == 3:
            continue

        y_shape = matching.get_y_shape()
        branch1, branch2 = matching.get_matching_branchs()
        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title('{i}th splitting point (type-{t}).'.format(i=id, t=matching.type))
        y_shape_branch1_nodes, y_shape_branch2_nodes = [node for node in y_shape['branch1']], [node for node in
                                                                                               y_shape['branch2']]
        yb1xs, yb1ys, yb1zs = zip(*y_shape_branch1_nodes)
        yb2xs, yb2ys, yb2zs = zip(*y_shape_branch2_nodes)
        ysx, ysy, ysz = y_shape['splitting_point']
        ax1.scatter(yb1xs, yb1ys, yb1zs, c='b', marker='o', s=10)
        ax1.scatter(yb2xs, yb2ys, yb2zs, c='b', marker='o', s=10, label='dm_source')
        ax1.scatter(ysx, ysy, ysz, c='r', s=40, label='source_splitting_point')
        for id, node in enumerate(y_shape_branch1_nodes):
            if id != 0:
                sx, sy, sz = y_shape_branch1_nodes[id - 1]
                ex, ey, ez = node
                ax1.quiver(
                    sx, sy, sz,
                    ex - sx, ey - sy, ez - sz,
                    color='b',
                )
        for id, node in enumerate(y_shape_branch2_nodes):
            if id != 0:
                sx, sy, sz = y_shape_branch2_nodes[id - 1]
                ex, ey, ez = node
                ax1.quiver(
                    sx, sy, sz,
                    ex - sx, ey - sy, ez - sz,
                    color='b',
                )
        if matching.type == 1:
            my_dict = matching.get_splitting_match()
            tid = list(my_dict)[0]
            sx, sy, sz = my_dict[tid]
            ax1.scatter(sx, sy, sz, c='c', s=30, label='matched_splitting_point')
        if len(branch1) > 0:
            # just draw one..
            branch1_nodes = list(branch1.values())[0]
            neuron_name = swc_file_names[list(branch1.keys())[0]]
            x, y, z = zip(*branch1_nodes)
            ax1.scatter(x, y, z, c='y', s=10, label=neuron_name)
            # ax.plot(x, y, z, c='y', )
        if len(branch2) > 0:
            # just draw one..
            branch2_nodes = list(branch2.values())[0]
            neuron_name = swc_file_names[list(branch2.keys())[0]]
            x, y, z = zip(*branch2_nodes)
            ax1.scatter(x, y, z, c='g', s=10, label=neuron_name)
            # ax.plot(x, y, z, c='g', )
        ax1.legend()

        ax2 = fig.add_subplot(122, projection='3d')
        for key, val in source_SWC.swc_dict.items():
            if val.father_id != -1:
                sx, sy, sz = val.get_coordinate()
                tx, ty, tz = source_SWC.swc_dict[val.father_id].get_coordinate()
                ax2.plot([sx, tx], [sy, ty], [sz, tz], 'b')
        ax2.scatter(ysx, ysy, ysz, c ='r', s=40)
        plt.show()


def output_matchings(matchings, filepath, swc_names):
    for i, matching in enumerate(matchings):
        if matching.type == 1 or matching.type == 2:

            y_shape_left_path = filepath.format(id = i, branch="y_shape_left")
            y_shape_right_path = filepath.format(id = i, branch="y_shape_right")
            matched_path = filepath.format(id = 1, branch="matched_branches")
            y_shape = matching.get_y_shape()
            left_branch = y_shape['branch1']
            right_branch = y_shape['branch2']
            left_branch_matched = matching.branch1_matching
            right_branch_matched = matching.branch2_matching
            # pdb.set_trace()
            with open(y_shape_left_path, 'w') as file:
                file.write("###Branch 1\n")
                for id, coord in enumerate(left_branch):
                    file.write(f"{id} 1 " + ' '.join(list(map(str,coord))) + f" 1 {id - 1}" + '\n')
                file.write("#Branch 2\n")
            with open(y_shape_right_path, 'w') as file:
                for id, coord in enumerate(right_branch):
                    file.write(f"{id} 1 " + ' '.join(list(map(str, coord))) + f" 1 {id - 1}" + '\n')
            if len(left_branch_matched) > 0:
                key  = list(left_branch_matched.keys())[0]
                value = left_branch_matched[key]
                match_left_branch_path = filepath.format(id=i, branch='matched_left_' + os.path.splitext(os.path.basename(swc_names[key]))[0])
                with open(match_left_branch_path, 'w') as file:
                    for id, coord in enumerate(value):
                        file.write(f"{id} 1 " + ' '.join(list(map(str, coord))) + f" 1 {id - 1}" + '\n')
            if len(right_branch_matched) > 0:
                key  = list(right_branch_matched.keys())[0]
                value = right_branch_matched[key]
                match_right_branch_path = filepath.format(id=i, branch='match_right_'+ os.path.splitext(os.path.basename(swc_names[key]))[0])
                with open(match_right_branch_path, 'w') as file:
                    for id, coord in enumerate(value):
                        file.write(f"{id} 1 " + ' '.join(list(map(str, coord))) + f" 1 {id - 1}" + '\n')



if __name__ == '__main__':
    summary_image_stack_folder = 'data/MouseLight/STP_180830_50um_Jai_summary/'
    mouselight_folder = 'data/MouseLight/target_neurons/'
    output_folder = 'data/MouseLight/output_decomposed/'
    dm_neuron_path = 'data/MouseLight/Jai_atlas_32_branch20.swc'
    output_branch_path = 'data/MouseLight/output_branches/50um_id_{id}_{branch}.swc'

    image_files = sorted(
        [os.path.join(summary_image_stack_folder, filename) for filename in os.listdir(summary_image_stack_folder)])
    image_stack = []
    for z, filepath in enumerate(image_files):
        image_stack.append(np.array(Image.open(filepath)).transpose())
        # image_node_list = (np.array(Image.open(filepath))
        # image_node_list = [(y, x, z) for (x, y) in image_node_list]
        # image_stack_node_set.update(image_node_list)
    summary_mat = np.array(image_stack)
    summary_mat = np.where(summary_mat > 0.1, 1, 0)
    SZ = len(image_files)
    SY, SX = np.array(Image.open(image_files[0])).shape

    mouselight_filepaths = [os.path.join(mouselight_folder, f) for f in os.listdir(mouselight_folder)]
    mouselight_SWCs = [SWCBranchingPoint(path) for path in mouselight_filepaths]
    mouselight_thicken_mats = [swc.get_thicken_mat((SX, SY, SZ), radius=5) for swc in mouselight_SWCs]

    dm_SWC = SWCBranchingPoint(dm_neuron_path)
    dm_SWC_thicken_mat = dm_SWC.get_thicken_mat((SX, SY, SZ), radius=5)

    masked_summary_mat = np.multiply(dm_SWC_thicken_mat, summary_mat)

    for i in range(dm_SWC_thicken_mat.shape[0]):
        cmat = masked_summary_mat[i,:]
        cimage = Image.fromarray(cmat)
        cimage.save('data/MouseLight/check_thicken_mat/image_{:04d}.tiff'.format(i), format='TIFF')

    mouselight_thicken_sample = mouselight_thicken_mats[0]
    for i in range(mouselight_thicken_sample.shape[0]):
        cmat = mouselight_thicken_sample[i, :]
        cimage = Image.fromarray(cmat)
        cimage.save('data/MouseLight/mouselight_thicken_mat/image_{:04d}.tiff'.format(i), format='TIFF')

    weights = construct_and_solve(mouselight_thicken_mats, masked_summary_mat)
    average_weight = sum(weights) / len(weights)
    selected_mouselight_indices = [i for i in range(len(weights)) if weights[i] > average_weight]

    # selected_mouselight_indices = [0, 1, 4, 6, 9]
    print('Selected mouselight indices: ', selected_mouselight_indices)
    selected_mouselight_SWCs = [mouselight_SWCs[i] for i in selected_mouselight_indices]
    selected_mouselight_filenames = [mouselight_filepaths[i] for i in selected_mouselight_indices]
    print(selected_mouselight_filenames)
    for swc in selected_mouselight_SWCs:
        swc.init_nearest_neighor_tree()
        swc.init_nearest_splitting_neighor_tree()

    dm_splitting_points = dm_SWC.search_for_splitting_point_by_hop2leaf()
    for swc in selected_mouselight_SWCs:
        swc._init_hop2leaf()
    # matchings = find_splitting_point_type(dm_SWC, dm_splitting_points, selected_mouselight_SWCs, )
    matchings = find_splitting_point_type_v1(dm_SWC, dm_splitting_points, selected_mouselight_SWCs, )
    for matching in matchings:
        print(matching.get_info_string())
    output_matchings(matchings, output_branch_path, selected_mouselight_filenames)
    # visualize_matchings(matchings, selected_mouselight_filenames, dm_SWC)
