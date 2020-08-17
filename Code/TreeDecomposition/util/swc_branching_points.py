from operator import attrgetter
import numpy as np
from sklearn.neighbors import NearestNeighbors


class TreeNode:
    def __init__(self, id, x, y, z, children, father_id, thickness = 0):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
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
    def __init__(self, filepath):
        self.swc_reader(filepath)

    def swc_reader(self, filepath):
        """
        :param filepath: filepath to swc file.
        """
        swc_dict = {}
        with open(filepath) as file:
            for line in file:
                line = line.strip()
                if not line.startswith('#'):
                    data = line.split()
                    x, y, z = int(float(data[2])), int(float(data[3])), int(float(data[4]))
                    thickness = float(data[5])
                    id, father_id = int(data[0]), int(data[6])
                    swc_dict[id] = TreeNode(id, x, y, z, [], father_id, thickness=thickness)
                    if father_id != -1:
                        swc_dict[father_id].children.append(id)
        self.swc_dict = swc_dict


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
        splitting_points = [node for node in self.swc_dict.values() if node.is_splitting_point()]
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

    def init_nearest_neighor_tree(self):
        self.ball_tree_points = list(self.swc_dict.values())
        all_points = [node.get_coordinate() for node in self.ball_tree_points]
        self.ball_tree = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(all_points)

    def init_nearest_splitting_neighor_tree(self):
        self.splitting_ball_tree_points = [node for node in self.swc_dict.values() if node.is_splitting_point()]
        all_splitting_points = [node.get_coordinate() for node in self.splitting_ball_tree_points]
        self.splitting_ball_tree = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(all_splitting_points)

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