from PIL import Image, ImageDraw
import os
import sys
from geojson import Feature, FeatureCollection, LineString
import geojson as gjson
from math import sqrt
from math import fabs


def read_ve(swc_file):
    nodes, edges = {}, []
    with open(swc_file) as file:
        for line in file:
            if not line.startswith('#'):
                data = line.strip().split()
                x, y, z = map(float, data[2:5])
                id, pid = int(data[0]), int(data[6])
                thickness = float(data[-2])
                nodes[id] = (x, y, z, thickness)
                if pid != -1:
                    edges.append((id, pid))
    return nodes, edges


# check if z-value is between the other two.
def in_between(z, uz, vz, eps=1e-6):
    max_uv = max(uz, vz)
    min_uv = min(uz, vz)
    return ((min_uv < z + 0.5)) and ((max_uv > z - 0.5))


# return segment on image z
def segment(u, v, z):
    if fabs(u[2] - v[2]) < 1e-5:
        # return (u[0]+(u[0] - v[0])/4, u[1]+(u[1] - v[1])/4, v[0]-(u[0] - v[0])/4, v[1]-(u[1] - v[1])/4)
        return (u[0], u[1], v[0], v[1])
    z_top = z - 0.5
    z_down = z + 0.5
    if u[2] > v[2]:
        u, v = v, u
    ru, rv = list(u), list(v)
    if u[2] < z_top:
        scale = (z_top - u[2]) / (v[2] - u[2])
        ru[0] = scale * (v[0] - u[0]) + u[0]
        ru[1] = scale * (v[1] - u[1]) + u[1]
    if v[2] > z_down:
        scale = (v[2] - z_down) / (v[2] - u[2])
        rv[0] = v[0] - scale * (v[0] - u[0])
        rv[1] = v[1] - scale * (v[1] - u[1])
    # return (ru[0]+(ru[0] - rv[0])/4, ru[1]+(ru[1] - rv[1])/4, rv[0]-(ru[0] - rv[0])/4, rv[1]-(ru[1] - rv[1])/4)
    return (ru[0], ru[1], rv[0], rv[1])

def project_point(u, v, z):
    if u[2] > v[2]:
        u, v = v, u
    x, y, eps = 0, 0, 1e-6
    if v[2] - u[2] > eps:
        scale = (z - u[2]) / (v[2] - u[2])
        x = scale * (v[0] - u[0]) + u[0]
        y = scale * (v[1] - u[1]) + u[1]
    else:
        x, y = (u[0] + v[0]) / 2, (u[1] + v[1]) / 2
    return (x, y)


def get_density(seg, cloud, radius=5.0):
    return 1
    midx = (seg[0] + seg[2]) / 2
    midy = (seg[1] + seg[3]) / 2
    area = radius ** 2
    density = 0.0
    for t in cloud:
        if (t[1] - midx)**2 + (t[0] - midy)**2 <= area:
            density += t[2]
    return density

def get_point_density(point, cloud, radius=5.0):
    return 1
    area = radius ** 2
    density = 0.0
    for t in cloud:
        if (t[1] - point[0]) ** 2 + (t[0] - point[1]) ** 2 <= area:
            density += t[2]
    return density


def get_all_project_points(nodes, edges, z_range):
    point_all = [[] for i in range(z_range)]
    max_density = 0.0
    for z in range(z_range):
        print(z)
        sys.stdout.flush()
        for e in range(len(edges)):
            edge = edges[e]
            u = nodes[edge[0]]
            v = nodes[edge[1]]
            if in_between(z, u[2], v[2]):
                point = project_point(u, v, z)
                density = get_point_density(point, cloud_list[z])
                point_all[z].append((point, density))
                max_density = max(max_density, density)
    return max_density, point_all


def get_all_segs(nodes, edges, z_range):
    seg_all = [[] for i in range(z_range+1)]
    max_density = 0.0
    for z in range(z_range):
        # print(z)
        sys.stdout.flush()
        for e in range(len(edges)):
            edge = edges[e]
            u = nodes[edge[0]]
            v = nodes[edge[1]]
            if in_between(z, u[2], v[2]):
                seg = segment(u, v, z)
                density = (u[3] + v[3]) / 2
                seg_all[z].append((seg, density, e))  # seg = (x1, y1, x2, y2), density, id(e))
                max_density = max(max_density, density)
    return max_density, seg_all


# Make an image at a certain Z-value and output.
def make_png(seg_all, z_range, dir_path, max_density, ind_arr, l=3000, w=2250, max_width=1):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    for z in range(z_range):
        # print("Generating " + str(z) + 'th image...')
        im = Image.new('I', (l, w), color=0)
        draw = ImageDraw.Draw(im)
        for seg in seg_all[z]:
            density = int(65535 * seg[1] / max_density)
            #print(seg, density)
            draw.line(tuple([x for x in seg[0]]), fill=density, width=int(1))
        ofilename = os.path.join(dir_path, './' + str(ind_arr[z-1]) + '.png')
        im.save(ofilename, format='PNG')



if __name__ == '__main__':
    swc_file = sys.argv[1]
    output_dir = swc_file[:-4]
    #z_range, width, length = 270, 855, 1137
    z_range, width, length = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    nodes, edges = read_ve(swc_file)

    # find where is the index
    ind_array = [i for i in range(z_range)]

    # get_all_segments
    print('Get segments')
    max_density, seg_all = get_all_segs(nodes, edges, z_range)
    print('Max density',max_density)
    # make pngs
    make_png(seg_all, z_range, output_dir, max_density, ind_array, l=length, w=width)




