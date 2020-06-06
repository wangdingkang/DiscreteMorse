import json
import os
import random
from math import sqrt
from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors

folder = 'MouseLight/jsons'
input_neurons = 'MouseLight/MouseLightMOinfo.txt'
image_stack_path = 'MouseLight/injection_atlas'

files = [os.path.join(folder, file) for file in os.listdir(folder)]
neuron_cloud =[]
neuron_ids = set()

def read_points(img_stack_folder):
    image_paths = os.listdir(img_stack_folder)
    z_range = range(1, len(image_paths) + 1)
    image_points, first_time = [], True
    for img_name, z in zip(image_paths, z_range):
        img_path = os.path.join(img_stack_folder, img_name)
        img = Image.open(img_path)
        img_arr = np.array(img)
        points = np.nonzero(img_arr > 128)
        z_axes = np.expand_dims(z * np.ones(len(points[0]), dtype=np.int64), axis=1)
        points = np.transpose(points)
        points = np.concatenate((points, z_axes), axis=1)
        image_points.extend(points.tolist())
    return image_points




if __name__ == '__main__':
    injection_nodes = read_points(image_stack_path)
    tree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(np.array(injection_nodes))

    with open(input_neurons) as file:
        for line in file:
            neuron_ids.add(line.strip())


    for file in files:
        with open(file) as json_file:
            data = json.load(json_file)
            neurons = data['neurons']
            for neuron in neurons:
                if neuron['idString'] in neuron_ids:
                    soma = neuron['soma']
                    x, y, z = soma['y']/50 + 10, soma['z']/50+10, soma['x']/50+10
                    neuron_cloud.append([neuron['idString'], x, y, z])
                    distances, indices = tree.kneighbors(np.array([[x, y, z]]))
                    if distances[0][0] < sqrt(3)/2:
                        print(neuron['idString'], distances[0][0])

