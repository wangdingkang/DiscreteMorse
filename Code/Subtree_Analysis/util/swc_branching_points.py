from operator import attrgetter
import numpy as np
from sklearn.neighbors import NearestNeighbors
import copy


class TreeNode:
    def __init__(self, id, x, y, z, children, father_id, thickness = 1):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.type = 1
        self.thickness = thickness
        self.children = children
        self.father_id = father_id
        self.importance = 0
        self.hop2leaf = 0

    def is_leaf(self):
        return len(self.children) == 0

    def is_splitting_point(self):
        return len(self.children) == 2

    def is_root(self):
        return self.father_id == -1

    def get_coordinate(self):
        return (self.x, self.y, self.z)


class SWCBranchingPoint:
    def __init__(self, filepath: str = None, dictionary: dict = None, offset: tuple = None, start_with_zero: bool = True):
        if filepath:
            self.swc_reader(filepath, offset, start_with_zero)
        else:
            self.swc_dict = dictionary

    def self_check(self):
        for key, value in self.swc_dict.items():
            print(key, value)

    def swc_reader(self, filepath, offset = None, start_with_zero=True):
        """
        :param filepath: filepath to swc file.
        """
        swc_dict = {}
        id_offset = 0 if start_with_zero else -1
        ox, oy, oz = 0, 0, 0
        if offset is not None:
            ox, oy, oz = offset
        print(filepath)
        with open(filepath) as file:
            for line in file:
                line = line.strip()
                if not line.startswith('#'):
                    data = line.split()
                    x, y, z = int(float(data[2])), int(float(data[3])), int(float(data[4]))
                    x = x + ox; y = y + oy; z = z + oz
                    thickness = float(data[5])
                    id, father_id = int(data[0]) + id_offset, int(data[6]) + id_offset
                    if father_id == -2:
                        father_id = -1
                    swc_dict[id] = TreeNode(id, x, y, z, [], father_id, thickness=thickness)
                    if father_id != -1:
                        swc_dict[father_id].children.append(id)
        self.swc_dict = swc_dict

    def get_all_subtrees(self):
        leaves = [node for node in self.swc_dict.values() if node.is_leaf()]
        cnt_leaves = len(leaves)
        print(f"We have {cnt_leaves} leaves.")
        if cnt_leaves > 15:
            print(f"The process will be slow if we have too many leaves.")
            exit(-1)
        i = 63
        for i in range(1, 2**cnt_leaves):
            selected_leaves = [leaves[j] for j in range(cnt_leaves) if (1 << j) & i != 0]
            sub_dict = self.get_subtree(selected_leaves)
            yield i, sub_dict

    def get_unmatched_subtree(self, matched_mask):
        leaves = [node for node in self.swc_dict.values() if node.is_leaf()]
        branches = {}
        for leaf_id, leaf in enumerate(leaves):
            if (1 << leaf_id) & matched_mask == 0:
                this_branch = self.get_branch_dict(leaf)
                branches = {**branches, **this_branch}
        return self.relabel_swc_dict(branches)

    def get_matched_subtree_v2(self, all_mouselight_coordinates, each_branch_try = 4):
        leaves = [node for node in self.swc_dict.values() if node.is_leaf()]
        mouselight_ball_tree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(all_mouselight_coordinates)
        cnt_leaves = len(leaves)
        # print(f"We have {cnt_leaves} leaves.")
        best_coords = [self.swc_dict[0].get_coordinate()]
        best_distance = self.get_incremental_matching_distance(all_mouselight_coordinates, best_coords, mouselight_ball_tree)
        best_dict = {0:copy.deepcopy(self.swc_dict[0])}
        best_match_mask = 0
        decrease_flag = True

        branches = {}
        branch_dicts = {}
        branch_tries = {}
        for leaf_id, leaf in enumerate(leaves):
            branch_dict = self.get_branch_dict(leaf)
            branch_keys = sorted(branch_dict.keys())
            this_branch_coordinates = [branch_dict[bkey].get_coordinate() for bkey in branch_keys]
            length = len(this_branch_coordinates)
            tries = [i * length // each_branch_try for i in range(1, each_branch_try)] + [len(this_branch_coordinates)]
            branches[leaf_id] = this_branch_coordinates
            branch_dicts[leaf_id] = branch_dict
            branch_tries[leaf_id] = tries
        while decrease_flag is True:
            # print(best_distance)
            decrease_flag = False
            temp_bid, temp_tid, temp_distance = -1, -1, best_distance
            for leaf_id, leaf in enumerate(leaves):
                if ((1<<leaf_id) & best_match_mask) == 0:
                    tries = branch_tries[leaf_id]
                    this_branch_coordinates = branches[leaf_id]
                    for id, t in enumerate(tries):
                        this_distance = self.get_incremental_matching_distance(all_mouselight_coordinates, list(set(best_coords+this_branch_coordinates[:t])), mouselight_ball_tree)
                        if this_distance < temp_distance:
                            decrease_flag = True
                            temp_distance = this_distance
                            temp_tid = id
                            temp_bid = leaf_id
            if temp_tid != -1 and temp_bid != -1:
                best_distance = temp_distance
                best_coords = list(set(best_coords + branches[temp_bid][:branch_tries[temp_bid][temp_tid]]))
                best_keys = sorted(branch_dicts[temp_bid].keys())
                best_selected_keys = best_keys[:branch_tries[temp_bid][temp_tid]]
                branch_dict = branch_dicts[temp_bid]
                selected_dict = {key : branch_dict[key] for key in best_selected_keys}
                best_dict = {** best_dict, ** selected_dict}
                if temp_tid == each_branch_try - 1:
                    best_match_mask = best_match_mask | (1<<temp_bid)
        return best_match_mask, best_distance, self.relabel_swc_dict(best_dict)

    def get_incremental_matching_distance(self, mouselight_points, subtree_points, mouselight_ball_tree ):
        subtree_ball_tree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(subtree_points)
        distance, _ = subtree_ball_tree.kneighbors(mouselight_points)

        totAB = 0.0
        # print(distance.shape)
        for i in range(len(mouselight_points)):
            totAB += distance[i, 0]
        dist_AB =  totAB / len(mouselight_points)

        totBA = 0.0
        distance, _ = mouselight_ball_tree.kneighbors(subtree_points)
        for i in range(len(subtree_points)):
            totBA += distance[i, 0]
        dist_BA =  totBA / len(subtree_points)
        return max(totAB, totBA)

    def get_branch_dict(self, leaf):
        ret_dict = {}
        temp = leaf
        while temp is not None:
            ret_dict[temp.id] = copy.deepcopy(temp)
            if temp.father_id != -1:
                temp = self.swc_dict[temp.father_id]
            else:
                break
        return ret_dict


    def get_subtree(self, leaves):
        ret_dict = {}
        for leaf in leaves:
            temp = leaf
            while temp is not None:
                ret_dict[temp.id] = copy.deepcopy(temp)
                if temp.father_id != -1:
                    temp = self.swc_dict[temp.father_id]
                else:
                    break
                # print(temp.id, temp.father_id)
        # print(len(ret_dict))
        return self.relabel_swc_dict(ret_dict)

    def relabel_swc_dict(self, arbitrary_dict):
        keys = sorted(arbitrary_dict.keys())
        keys = [-1] + keys
        key_mapping = {}
        for id, key in enumerate(keys):
            key_mapping[key] = id - 1
        sorted_dict = {key_mapping[key] : arbitrary_dict[key] for key in arbitrary_dict}
        for key in sorted_dict:
            father_id = sorted_dict[key].father_id
            sorted_dict[key].id = key
            sorted_dict[key].father_id = key_mapping[father_id]
        return sorted_dict

    def _init_hop2leaf(self):
        leaves = [node for node in self.swc_dict.values() if node.is_leaf()]
        for leaf in leaves:
            hop, current_node = 0, leaf
            while not current_node.is_root():
                if current_node.hop2leaf < hop or current_node.is_leaf():
                    current_node.hop2leaf = hop
                    current_node = self.swc_dict[current_node.father_id]
                    hop += 1
                else:
                    break

    def search_for_splitting_point_by_hop2leaf(self, topk=15):
        """
        :param topk: how many top important splitting node to keep
        :return: a list of top k splitting nodes.
        """
        self._init_hop2leaf()
        splitting_points = [node for node in self.swc_dict.values() if node.is_splitting_point() and not node.is_root()]
        for splitting_point in splitting_points:
            children = splitting_point.children
            splitting_point.importance = self.swc_dict[children[0]].hop2leaf
            for child in children:
                splitting_point.importance = min(splitting_point.importance, self.swc_dict[child].hop2leaf)
        splitting_points.sort(key=attrgetter('importance'), reverse=True)
        return splitting_points[:topk] if len(splitting_points) > topk else splitting_points

    def search_for_all_splitting_point(self):
        """
        :return: Just return all splitting points.
        """
        splitting_points = [node for node in self.swc_dict.values() if node.is_splitting_point()]
        return splitting_points

    def __valid(self, dim, coord):
        cx, cy, cz = coord
        sx, sy, sz = dim
        return 0 <= cx < sx and 0 <= cy < sy and 0 <= cz < sz

    def squared_distance2segment(self, ep1, ep2, p):
        """
        :param ep1: end point 1
        :param ep2: end point 2
        :param p: the point p,
        :return: return distance from p to the SEGMENT (ep1, ep2)
        """
        epx1, epy1, epz1 = ep1
        epx2, epy2, epz2 = ep2
        px, py, pz = p
        squared_length = (epx1 - epx2) ** 2 + (epy1 - epy2) ** 2 + (epz1 - epz2) ** 2
        if squared_length < 1e-6:
            return (px - epx2) ** 2 + (py - epy2) ** 2 + (pz - epz2) ** 2
        t = max(0, min(1, float(
            (px - epx1) * (epx2 - epx1) + (py - epy1) * (epy2 - epy1) + (pz - epz1) * (epz2 - epz1)) / squared_length))
        tx, ty, tz = epx1 + t * (epx2 - epx1), epy1 + t * (epy2 - epy1), epz1 + t * (epz2 - epz1)
        return (px - tx) ** 2 + (py - ty) ** 2 + (pz - tz) ** 2

    def get_thicken_mat(self, dim, radius=5, method='edge_wise'):
        """
        :param dim: size of the brain space.
        :param radius: thickening radius
        :param method: how to thicken, edge_wise, node_wise?
        :return: a mask 3D mat with shape = dim. 1 for foreground and 0 for background.
        """
        SX, SY, SZ = dim
        img_array = np.zeros((SZ, SX, SY), dtype=int)
        treenodes = self.swc_dict.values()
        if method == 'edge_wise':
            for treenode in treenodes:
                if not treenode.is_root():
                    x1, y1, z1 = treenode.x, treenode.y, treenode.z
                    another_treenode = self.swc_dict[treenode.father_id]
                    x2, y2, z2 = another_treenode.x, another_treenode.y, another_treenode.z
                    minx, maxx = max(0, min(x1, x2) - radius), min(SX, (max(x1, x2) + radius))
                    miny, maxy = max(0, min(y1, y2) - radius), min(SY, (max(y1, y2) + radius))
                    minz, maxz = max(0, min(z1, z2) - radius), min(SX, (max(z1, z2) + radius))
                    for nx in range((minx), (maxx)):
                        for ny in range((miny), (maxy)):
                            for nz in range((minz), (maxz)):
                                sdist = self.squared_distance2segment((x1, y1, z1), (x2, y2, z2), (nx, ny, nz))
                                if sdist <= radius ** 2:
                                    img_array[nz, nx, ny] = 1

        else:
            for treenode in treenodes:
                x, y, z = treenode.x, treenode.y, treenode.z
                for dx in range(-radius, radius):
                    for dy in range(-radius, radius):
                        for dz in range(-radius, radius):
                            if dx ** 2 + dy ** 2 + dz ** 2 <= radius ** 2:
                                tx, ty, tz = (x + dx), (y + dy), (z + dz)
                                if self.__valid((SZ, SX, SY), (tz, tx, ty)):
                                    img_array[tz, tx, ty] = 1
        return img_array

    def __get_upbranch(self, starting_point, hop_limit=20):
        ret, temp = [], starting_point
        while hop_limit > 0:
            if temp.father_id != -1:
                temp = self.swc_dict[temp.father_id]
            else:
                break
            if len(ret) == 0 or temp.get_coordinate() != ret[-1]:
                ret.append(temp.get_coordinate())
                hop_limit -= 1
        ret.reverse()
        return ret

    def __get_branch(self, starting_point, hop_limit=20):
        ret, temp = [], starting_point
        while hop_limit > 0:
            if len(ret)==0 or temp.get_coordinate() != ret[-1]:
                ret.append(temp.get_coordinate())
                hop_limit -= 1
            temp_nexts = temp.children
            if len(temp_nexts) == 0:
                break
            for next in temp_nexts:
                if self.swc_dict[next].hop2leaf == temp.hop2leaf - 1:
                    temp = self.swc_dict[next]
                    continue
            # import pdb
            # if temp.id == 4232:
            #     pdb.set_trace()
        return ret


    def get_single_branch(self, node_id: int, hop_limit=20):
        starting_point = self.swc_dict[node_id]
        upstream_branch = self.__get_upbranch(starting_point, hop_limit)
        downstream_branch = self.__get_branch(starting_point, hop_limit)
        return upstream_branch + downstream_branch[1:]

    def get_split_branches(self, splitting_node_id: int, hop_limit=20):
        """
        :param splitting_node_id: starting point id
        :param hop_limit: length limit of the returned branch
        :return: a dict of three lists of (x, y, z) node list based on hop....
        """
        starting_point = self.swc_dict[splitting_node_id]
        assert (starting_point.is_splitting_point())
        upstream_branch = self.__get_upbranch(starting_point, hop_limit)
        branch1, branch2 = [starting_point.get_coordinate()], [starting_point.get_coordinate()]
        temp1, temp2 = self.swc_dict[starting_point.children[0]], self.swc_dict[starting_point.children[1]]
        branch1.extend(self.__get_branch(temp1, hop_limit - 1))
        branch2.extend(self.__get_branch(temp2, hop_limit - 1))
        return {'splitting_point': starting_point.get_coordinate(), 'branch1': upstream_branch + branch1,
                'branch2': upstream_branch + branch2}

    def init_nearest_neighor_tree(self, n_neighbors = 5):
        self.ball_tree_points = list(self.swc_dict.values())
        all_points = [node.get_coordinate() for node in self.ball_tree_points]
        self.ball_tree = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(all_points)

    def init_nearest_splitting_neighor_tree(self, n_neighbors=5):
        self.splitting_ball_tree_points = [node for node in self.swc_dict.values() if node.is_splitting_point()]
        all_splitting_points = [node.get_coordinate() for node in self.splitting_ball_tree_points]
        self.splitting_ball_tree = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(all_splitting_points)

    def overlap_with_branch(self, branch, dist_bound=5, coverage_bound=0.75):
        """
        :param branch: A branch from the DM tree.
        :param dist_bound: max distance allowed for any branch node.
        :param coverage_bound: pass the ratio bound of nodes coverage, then consider it as a match.
        :return: boolean if this branch is contained. And a list of nodes for matched points in this neuron.
        """
        tot_num, coverage_num = len(branch), 0
        assert (self.ball_tree is not None)
        distances, indices = self.ball_tree.kneighbors(branch)
        coords = [self.ball_tree_points[ns[0]].get_coordinate() for ns in indices.tolist()]
        coverage_num = len(np.argwhere(distances <= dist_bound))
        return coverage_num >= coverage_bound * tot_num, coords

    def matching_splitting_point(self, source_splitting_coords, dist_bound = 5):
        distance, index = self.splitting_ball_tree.kneighbors([source_splitting_coords])
        coord = self.splitting_ball_tree_points[index[0][0]].get_coordinate()
        return distance[0,0] < dist_bound, coord

    def matching_point(self, source_point_coords, dist_bound=5):
        distance, index = self.ball_tree.kneighbors([source_point_coords])
        coord = self.ball_tree_points[index[0][0]].get_coordinate()
        return distance[0, 0] < dist_bound, coord

    def get_matched_splitting_point(self, source_splitting_coords):
        distance, index = self.splitting_ball_tree.kneighbors([source_splitting_coords])
        indexes = index[0]
        close_splitting_points = [self.splitting_ball_tree_points[i] for i in indexes]
        return close_splitting_points

    def get_matched_point(self, source_coords):
        distance, index = self.ball_tree.kneighbors([source_coords])
        indexes = index[0]
        close_points = [self.ball_tree_points[i] for i in indexes]
        return close_points

    def get_matched_distance(self, source_coords):
        distance, _ = self.ball_tree.kneighbors(source_coords)
        tot = 0.0
        # print(distance.shape)
        for i in range(len(source_coords)):
            tot += distance[i, 0]
        return tot / len(source_coords)

    def get_all_coordinates(self):
        coord = []
        for val in self.swc_dict.values():
            coord.append(val.get_coordinate())
        return coord

    def get_direction(self, node, hops = 5):
        temp = node
        direction = [0., 0., 0.]
        cnt = 0
        while cnt < hops:
            father_id = temp.father_id
            if father_id == -1:
                break
            father = self.swc_dict[temp.father_id]
            coord_temp = temp.get_coordinate()
            coord_father = father.get_coordinate()
            dx, dy, dz = [coord_temp[i] - coord_father[i] for i in range(3)]
            cadin = np.sqrt(dx**2 + dy**2 + dz**2)
            dx, dy, dz = dx / cadin, dy / cadin, dz / cadin
            direction[0] += dx; direction[1] += dy; direction[2] += dz
            temp = father
            cnt += 1
        if cnt > 0:
            direction = [d / cnt for d in direction]
        return direction

    def get_leaves_and_directions(self):
        leaves = [node for node in self.swc_dict.values() if node.is_leaf()]
        directions = []
        for node in leaves:
            directions.append(self.get_direction(node))
        return leaves, directions

    def get_only_in_bound_subtree(self, bounds):
        ret_dict, starting_root_id = {}, 0
        self.recursive_search(ret_dict, starting_root_id, bounds=bounds)
        return self.relabel_swc_dict(ret_dict)

    def recursive_search(self, ret_dict, temp_id, bounds):
        ret_dict[temp_id] = self.swc_dict[temp_id]
        lx, rx, ly, ry, lz, rz = bounds
        # import pdb
        # pdb.set_trace()
        # print(self.swc_dict[temp_id].children)
        for child_id in self.swc_dict[temp_id].children:
            (x, y, z) = self.swc_dict[child_id].get_coordinate()
            # print(x, y, z, dx, dy, dz)
            # print(x, y, z)
            if lx < x < rx and ly < y < ry and lz < z < rz:
                self.recursive_search(ret_dict, child_id, bounds)

    def write_to_file(self, filepath):
        keys = sorted(self.swc_dict.keys())
        # print(len(keys))
        with open(filepath, 'w') as file:
            for key in keys:
                tree_node = self.swc_dict[key]
                file.write(f"{key} {tree_node.type} {tree_node.x} {tree_node.y} {tree_node.z} {tree_node.thickness} {tree_node.father_id}" + "\n")

    def get_distance(self, node_main:TreeNode, node_b:TreeNode, main_offset = (0,0, 0)):
        x1, y1, z1 = node_main.get_coordinate()
        ox, oy, oz = main_offset
        x1 -= ox; y1 -= oy; z1 -= oz
        x2, y2, z2 = node_b.get_coordinate()
        return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

    def concat_branch(self, offset, another_swc):
        the_other_root = another_swc.swc_dict[0]
        leaves = [node for node in self.swc_dict.values() if node.is_leaf()]
        ox, oy, oz = offset
        min_dist, min_index = 1e6, -1
        for id, node in enumerate(leaves):
            dist = self.get_distance(node, the_other_root, offset)
            if dist < min_dist:
                min_dist, min_index = dist, id
        concat_id = leaves[min_index].id
        id_offset = len(self.swc_dict)
        other_swc_keys = sorted(another_swc.swc_dict.keys())
        self.swc_dict[concat_id].children = [id_offset]
        for value in another_swc.swc_dict.values():
            value.x += ox; value.y += oy; value.z += oz
        for key in other_swc_keys:
            node = another_swc.swc_dict[key]
            old_id, old_father_id = node.id, node.father_id
            new_id = old_id + id_offset
            new_father_id = concat_id if old_father_id == -1 else old_father_id + id_offset
            node.id = new_id; node.father_id = new_father_id
            self.swc_dict[new_id] = node

    def translate(self, X, Y, Z):
        for value in self.swc_dict.values():
            value.x += X; value.y += Y; value.z += Z

