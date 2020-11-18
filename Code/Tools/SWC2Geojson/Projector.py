from PIL import Image, ImageDraw
import os
import sys
from geojson import Feature, FeatureCollection, LineString
import geojson as gjson
from math import sqrt
from math import fabs

cloud_list = None
# VX, VY, VZ = 10, -10, 1
VX, VY, VZ = 1, 1, 1

# Read in file.
def read_all(filename):
    nodes, edges = [], []
    with open(filename) as file:
        s = file.readline()
        n = int(s.strip())
        for i in range(n):
            s = file.readline()
            cs = [float(x) for x in s.strip().split()[:3]]
            cs[0], cs[1] = cs[1], cs[0]
            nodes.append(cs)
        for i in range(n - 1):
            s = file.readline()
            cs = tuple([int(x) for x in s.strip().split()[:2]])
            edges.append(cs)
            if len(edges) % 100000 == 0:
                print(len(edges))
                sys.stdout.flush()
    return nodes, edges

def read_ve(vfilename, efilename):
    nodes, edges = [], []
    with open(vfilename) as file:
        for line in file:
            node = [float(x) for x in line.strip().split()[:3]]
            node[0], node[1] = node[1] * VX, node[0] * VY
            node[2] = node[2] * VZ
            node = tuple(node)
            nodes.append(node)
    with open(efilename) as file:
        for line in file:
            edge = tuple([int(x) for x in line.strip().split()[:2]])
            edges.append(edge)
            if len(edges) % 100000 == 0:
                print(len(edges))
                sys.stdout.flush()
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
    print(len(edges), len(nodes))
    seg_all = [[] for i in range(z_range+1)]
    max_density = 0.0
    for z in range(z_range):
        print(z)
        sys.stdout.flush()
        for e in range(len(edges)):
            edge = edges[e]
            u = nodes[edge[0]]
            v = nodes[edge[1]]
            if in_between(z, u[2], v[2]):
                seg = segment(u, v, z)
                density = 1
                # density = get_density(seg, cloud_list[z])
                seg_all[z].append((seg, density, e))  # seg = (x1, y1, x2, y2), density, id(e))
                max_density = max(max_density, density)
    return max_density, seg_all


# make image
def make_png_point(point_all, z_range, dir_path, max_density, l=3000, w=2250, max_width=1):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    for z in range(z_range):
        print("Generating " + str(z) + 'th image...')
        im = Image.new('L', (l, w), color=0)
        draw = ImageDraw.Draw(im)
        # for point in point_all[z]:
        #     draw.point(point, fill=255)

        for point in point_all[z]:
            midx = point[0][0]
            midy = point[0][1]
            rad = int(max_width*sqrt(point[1]/max_density))
            draw.ellipse([midx - rad/2, midy - rad/2, midx + rad/2, midy + rad/2], fill=255)
        ofilename = os.path.join(dir_path, './' + str(z) + '.png')
        im.save(ofilename, format='png')


# Make an image at a certain Z-value and output.
def make_png(seg_all, z_range, dir_path, max_density, l=3000, w=2250, max_width=1):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    for z in range(z_range):
        print("Generating " + str(z) + 'th image...')
        im = Image.new('L', (l, w), color=0)
        draw = ImageDraw.Draw(im)
        for seg in seg_all[z]:
            draw.line(tuple([x for x in seg[0]]), fill=255, width=int(1))
        ofilename = os.path.join(dir_path, './{:04d}.png'.format(z))
        im.save(ofilename, format='png')
    # im = Image.new('L', (l, w), color=0)
    # draw = ImageDraw.Draw(im)
    # for edge in edges:
    #     u = nodes[edge[0]]
    #     v = nodes[edge[1]]
    #     if in_between(z, u[2], v[2]):
    #         seg = segment(u, v, z)
    #         draw.line(segment(u, v, z), fill=255, width=get_linewidth(seg, cloud_list[z], max_density))
    # return im

def make_geojson(seg_all, z_range, dir_path, max_density, ind_array=None, scale=1, max_width=10):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    if ind_array is None:
        ind_array = [i for i in range(z_range)]
    for z in range(z_range):
        features = []
        json_filename =  '{:04d}.json'.format(ind_array[z-1])
        output_file = os.path.join(dir_path, json_filename)
        for seg in seg_all[z]:
            seg_rescale = [x * scale for x in seg[0]]
            features.append(Feature(id=seg[2], geometry=LineString([(seg_rescale[0], seg_rescale[1]), (seg_rescale[2], seg_rescale[3])]),
                                    properties={"stroke-width": 1}))
        with open(output_file, 'w') as file:
            file.write(gjson.dumps(FeatureCollection(features), sort_keys=True))


def find_index_from_filename(filelist):
    pieces = filelist[0].split('_')
    len_piece = len(pieces)
    ind_piece = 0
    for i in range(len_piece):
        if pieces[-(i+1)].isdigit():
            ind_piece = -(i+1)
            break
    return ind_piece


def read_in_cloud(filename, z_range):
    cloud = [[] for i in range(z_range + 1)]
    with open(filename, 'r') as file:
        for line in file:
            x, y, z, density = line.strip().split()
            x, y, z, density = int(x), int(y), int(z), float(density)
            if z <= z_range:
                cloud[z].append((x, y, density))
    return cloud


if __name__ == '__main__':
    argc = len(sys.argv)
    dir_name = ''
    nodes, edges = [], []
    z_range = 0
    flag_image = True
    cloud_file = ''

    if argc != 6:
        print('The input should be [input_vert] [input_edge] [max z] [length] [width].')
        exit(0)
    elif argc == 5: # only one file with both vert & edges.
        file_vert = sys.argv[1]
        file_edge = sys.argv[2]
        dir_name = os.path.dirname(file_vert)
        z_range = int(sys.argv[3])
        length, width = int(sys.argv[4]), int(sys.arbv[5])
        print(file_vert, file_edge, z_range, length, width)
        sys.stdout.flush()
        nodes, edges = read_ve(file_vert, file_edge)

    # filename = sys.argv[1]
    # image_folder = './' + filename
    # onlyfiles = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    # # find where is the index
    # print('find index')
    # sys.stdout.flush()
    # ind = find_index_from_filename(onlyfiles)
    # print('Index found')
    # sys.stdout.flush()
    # ind_array = [s.split('_')[ind] for s in onlyfiles]
    # ind_array.sort()

    # read_in cloud points
    # print('Read cloud')
    # sys.stdout.flush()
    # cloud_list = read_in_cloud(cloud_file, z_range)

    # get_all_segments
    print('Get segments')
    sys.stdout.flush()
    max_density, seg_all = get_all_segs(nodes, edges, z_range)

    # get all project points
    # print('Get points')
    # sys.stdout.flush()
    # max_density, point_all = get_all_project_points(nodes, edges, z_range)


    # make pngs
    # image_path_dot = os.path.join(dir_name, 'Projection_dots')
    image_path_seg = os.path.join(dir_name, 'Projection_segs')
    # make_png_point(point_all, z_range, image_path_dot, max_density)
    make_png(seg_all, z_range, image_path_seg, max_density, l=length, w=width)

    # make_geojsons
    # print('jsoning')
    # sys.stdout.flush()
    # output_json = os.path.join(dir_name, 'GeoJson')
    # make_geojson(seg_all, z_range, output_json, max_density)



