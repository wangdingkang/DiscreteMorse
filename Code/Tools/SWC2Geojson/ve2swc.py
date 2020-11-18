import os
import sys

input_vert = sys.argv[1] # x y z
input_edge = sys.argv[2] # u -> v u < v

# input_vert = 'STP_180830/stree_v0.txt'
# input_edge = 'STP_180830/stree_e0.txt'
output_swc = os.path.join(os.path.dirname(input_vert), 'neuron.swc')


class swc:

    def __init__(self, x, y, z, thickness, index):
        self.x = x
        self.y = y
        self.z = z
        self.thickness = thickness
        self.index = index
        # self.r = '1'
        self.type = '0'
        self.parent = '-1'

    def set_parent(self, parent):
        self.parent = parent

    # flip x and y.
    def to_string(self):
        return ' '.join([self.index, self.type, self.y, self.x, self.z, self.thickness, self.parent])




if __name__ == '__main__':
    swc_nodes, cnt = [], 0
    with open(input_vert, 'r') as file:
        for line in file:
            x, y, z = line.strip().split()[:3]
            swc_nodes.append(swc(x, y, z, thickness='1', index=str(cnt)))
            cnt += 1

    with open(input_edge, 'r') as file:
        for line in file:
            u, v = line.strip().split()[:2]
            swc_nodes[int(v)].set_parent(u)

    with open(output_swc, 'w') as file:
        for node in swc_nodes:
            file.write(node.to_string() + '\n')