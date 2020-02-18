import os
import sys
import argparse
from math import sqrt
import networkx as nx

parser = argparse.ArgumentParser()
parser.add_argument('dir_name', help=r"Path to the directory of other files (vert.txt, edge.txt, multi_roots.txt).")
args = parser.parse_args()

# input vert.txt and edge.txt
dirname = args.dir_name
input_vert = dirname + 'vert.txt'
input_edge = dirname + 'edge.txt'

# positions of roots
input_pos = dirname + 'multi_roots.txt'

# prefix of outputs
output_vert = dirname + 'stree_v'
output_edge = dirname+ 'stree_e'
output_extension = '.txt'

def get_dist(u, v, favg, cavg):
    return sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2 + (u[2] - v[2])**2) / (favg + 1)


def get_root_id(root, verts_list):
    min_dist, rid = 1e10, -1
    for id, v in enumerate(verts_list):
        tdist = get_dist(v, root, 0, 0)
        if tdist < min_dist:
            min_dist = tdist
            rid = id
    return rid


def output(tree_cnt, graph):
    edges = list(graph.edges())
    nodes = list(graph.nodes())
    edges.sort()
    nodes.sort()
    print(len(nodes), 'nodes in the', tree_cnt, 'tree.')
    output_vp = output_vert + str(tree_cnt) + output_extension
    output_ep = output_edge + str(tree_cnt) + output_extension
    with open(output_vp, 'w') as file:
        for i in nodes:
            pos = graph.nodes[i]['pos']
            dist = graph.nodes[i]['dist']
            file.write('{d[0]} {d[1]} {d[2]}'.format(d=pos) + ' ' + str(dist) + '\n')
    with open(output_ep, 'w') as file:
        for e in edges:
            file.write('{} {} {} {} {} {}\n'.format(e[0], e[1], graph.nodes[e[0]]['f'],graph.nodes[e[1]]['f'], graph.nodes[e[0]]['c'], graph.nodes[e[1]]['c']))

    
if __name__ == '__main__':

    roots, root_ids = [], []
    verts = []
    # read in input roots
    with open(input_pos, 'r') as file:
        for line in file:
            if not line.startswith('#'):
                roots.append(list(map(int, line.strip().split())))

    G = nx.Graph()
    cnt = 0
    with open(input_vert, 'r') as file:
        for line in file:
            x, y, z = list(map(int, line.strip().split()[:3]))
            # verts.append((x, y, z))
            G.add_node(cnt)
            verts.append([x, y, z, 0, 0])
            cnt += 1

    temp_edge = []
    with open(input_edge, 'r') as file:
        vert_attrs = {}
        for line in file:
            # print(line)
            u, v, f0, f1, c0, c1 = 0, 0, 1, 1, 1, 1
            data = line.strip().split()
            length = len(data)
            u, v = int(data[0]), int(data[1])
            if length == 6:
                f0, f1, c0, c1 = list(map(float, line.strip().split()[2:]))
            temp_edge.append((u, v))
            verts[u][3], verts[u][4] = max(verts[u][3], f0), max(verts[u][4], c0)
            verts[v][3], verts[v][4] = max(verts[v][3], f1), max(verts[v][4], c1)
    for e in temp_edge:
        u, v = e
        favg = (verts[u][3] + verts[v][3]) / 2
        cavg = (verts[u][4] + verts[v][4]) / 2
        # print(u, v, get_dist(verts[u], verts[v], favg, cavg))
        G.add_edge(u, v, weight=get_dist(verts[u], verts[v], favg, cavg))

    for root in roots:
        id = get_root_id(root, verts)
        root_ids.append(id)

    print(len(roots), ' roots, ', len(root_ids), ' points labelled.')

    error = 1e-6
    final_pred, final_dist = None, None
    for root_id in root_ids:
        pred, dist = nx.dijkstra_predecessor_and_distance(G, root_id)
        # print(pred)
        if final_pred is None:
            final_pred, final_dist = pred, dist
        else:
            # merge two distances & predecessor arrays.
            for i in range(cnt):
                if i in dist.keys():
                    if i not in final_dist.keys():
                        final_dist[i], final_pred[i] = dist[i], pred[i]
                    elif dist[i] < final_dist[i] + error:
                        final_pred[i], final_dist[i] = pred[i], dist[i]

    # Now, we get the final_pred and final_dist array, think about how to output them into relabelled multiple trees.
    # Build a new forest graph GF.
    GF = nx.Graph()
    for i in range(cnt):
        if i in final_dist.keys():
            GF.add_node(i, pos=verts[i][:3], dist=final_dist[i], f=verts[i][3], c=verts[i][4])

    for id, pre in final_pred.items():
        if len(pre) != 0:
            edge_data = G.get_edge_data(pre[0], id)
            GF.add_edge(pre[0], id, **edge_data)

    tree_n = 0
    for root_id in root_ids:
        component = GF.subgraph(nx.node_connected_component(GF, root_id))
        nodes = list(component.nodes)
        nodes.remove(root_id)
        mapping = dict(zip(sorted(nodes), range(1, len(component.nodes))))
        mapping[root_id] = 0
        component = nx.relabel_nodes(component, mapping, copy=True)
        output(tree_n, component)
        tree_n += 1
