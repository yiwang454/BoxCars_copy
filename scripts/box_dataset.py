import _init_paths

import json

from boxcars_dataset import BoxCarsDataset
from boxcars_data_generator import BoxCarsDataGenerator

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dataset = BoxCarsDataset(load_split="hard", load_atlas=True)
DATA_ROOT = "/home/vivacityserver6/datasets/BoxCars116k/"

import uuid
import cv2
import numpy as np
import os
from PIL import Image


def draw_bb2d(img, bb2d):
    x = bb2d[0]
    y = bb2d[1]
    w = bb2d[2]
    h = bb2d[3]
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 1)

    c = (int(x + w / 2), int(y + h / 2))
    return img, c


def draw_bb3d(img, bb3d):
    c = (255, 255, 0)
    for i in range(8):
        x = int(bb3d[i, 0])
        y = int(bb3d[i, 1])
        cv2.circle(img, (x, y), 1, c, -1)
    return img


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return (int(x), int(y))
    else:
        return False


def get_angle_from_two_points(p1, p2, deg=True):
    angle = np.arctan((p2[1] - p1[1]) / (1e-3 + (p2[0] - p1[0])))
    if deg:
        return angle * 180 / np.pi
    else:
        return angle


def cross_from_points(bb3d, img=None):
    front_lines = [(0, 5, 1, 4), (1, 6, 2, 5), (0, 2, 1, 3)]
    back_lines = [(3, 6, 2, 7), (3, 4, 0, 7), (4, 6, 5, 7)]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    angles = []
    for f, b, color in zip(front_lines, back_lines, colors):
        L1 = line(bb3d[f[0], :], bb3d[f[1], :])
        L2 = line(bb3d[f[2], :], bb3d[f[3], :])
        R12 = intersection(L1, L2)

        L3 = line(bb3d[b[0], :], bb3d[b[1], :])
        L4 = line(bb3d[b[2], :], bb3d[b[3], :])
        R34 = intersection(L3, L4)

        if img is not None:
            cv2.line(img, R12, R34, color, 1)
        angles.append(get_angle_from_two_points(R12, R34))

    if img is not None:
        return img, angles
    else:
        return angles


def draw_bbs(vehicle_id, instance):
    img = dataset.get_image(vehicle_id, instance)

    vehicle, instance, bb3d = dataset.get_vehicle_instance_data(
        vehicle_id, instance)

    bb2d = instance["2DBB"]
    img, c = draw_bb2d(img, bb2d)

    img = draw_bb3d(img, bb3d)

    img, angles = cross_from_points(bb3d, img)
    plt.imshow(img)
    plt.show()

try:
    os.mkdir('/home/vivacityserver6/datasets/BoxCars116k/anns_val/')
    os.mkdir('/home/vivacityserver6/datasets/BoxCars116k/ims_val/')
except:
    pass

VEHICLE_TYPES = {
    "combi": 0,
    "hatchback": 1,
    "sedan": 2,
    "suv": 3,
    "van": 4,
    "mpv": 5,
    "coupe": 6,
    "cabriolet": 7,
    "pickup": 8,
    "fastback": 9,
    "offroad": 10
}


def parse_vehicle(vehicle_id):
    vehicle, instance_, bb3d_ = dataset.get_vehicle_instance_data(
        vehicle_id, 0)

    vehicle_class = vehicle["annotation"]
    brand, model, vehicle_type, mk = vehicle_class.split(' ')
    to_camera = vehicle["to_camera"]
    if to_camera:
        to_camera = 1
    else:
        to_camera = 0

    for i in range(len(vehicle["instances"])):
        instance = vehicle["instances"][i]
        bb2d = instance["2DBB"]

        bb3d = instance["3DBB"]
        angles = cross_from_points(bb3d)

        while True:
            filename = str(uuid.uuid4())
            if not os.path.isfile("{0}/anns/{1}.txt".format(
                    DATA_ROOT, filename)):
                break
        with open("{0}/anns/{1}.txt".format(DATA_ROOT, filename), 'w') as f:
            f.write("{0} {1} {2} {3} {4}".format(VEHICLE_TYPES[vehicle_type],
                                                 angles[0], angles[1],
                                                 angles[2], to_camera))

        img = dataset.get_image(vehicle_id, i)
        img_crop = img[bb2d[1]:bb2d[1] + bb2d[3], bb2d[0]:bb2d[0] + bb2d[2], :]
        im = Image.fromarray(img_crop)
        im.save("{0}/ims/{1}.png".format(DATA_ROOT, filename))

def parse_vehicle_filter(vehicle_id, folder):
    vehicle, instance_, bb3d_ = dataset.get_vehicle_instance_data(
        vehicle_id, 0)

    vehicle_class = vehicle["annotation"]
    # brand, model, vehicle_type, mk = vehicle_class.split(' ')
    to_camera = vehicle["to_camera"]
    if to_camera:
        to_camera = 1
    else:
        to_camera = 0

    for instance_id in range(len(vehicle["instances"])):
        instance = vehicle["instances"][instance_id]
        bb2d = instance["2DBB"]
        bb3d = instance["3DBB"]
        offset = instance["3DBB_offset"]
        angles = cross_from_points(bb3d)
        ar = bb2d[3] / bb2d[2]
        aspect_judge = not (abs(ar-1.0) < 0.1 and abs(angles[0]) < 50*np.pi/180)
        if aspect_judge and (bb2d[2] > 50 and bb2d[3] > 50): 
            
            filename = str(vehicle_id) + '_' + str(instance_id)
            
            info = {'bb2d' : bb2d.tolist(), 'bb3d' : bb3d.tolist(), '3d_offset': offset.tolist()}
            info['class'] = vehicle_class
            info['to_camera'] = to_camera
            with open("{}/anns_val/{}/{}.txt".format(DATA_ROOT, folder, filename), 'w') as file:
                json.dump(info, file, indent=4)

            img = dataset.get_image(vehicle_id, instance_id)
            img_crop = img[bb2d[1]:bb2d[1] + bb2d[3], bb2d[0]:bb2d[0] + bb2d[2], :]
            im = Image.fromarray(img_crop)
            im.save("{}/ims_val/{}/{}.png".format(DATA_ROOT, folder, filename))

        # # for comparison purpose
        # if aspect_judge: 
            
        #     filename = str(vehicle_id) + '_' + str(instance_id)
            
        #     img = dataset.get_image(vehicle_id, instance_id)
        #     img_crop = img[bb2d[0]:bb2d[0] + bb2d[3], bb2d[1]:bb2d[1] + bb2d[2], :]
        #     im = Image.fromarray(img_crop)
        #     im.save("{0}/comparison/{1}.png".format(DATA_ROOT, filename))


all_vehicles = len(dataset.dataset["samples"])

dataset.initialize_data("validation")

part_data = dataset.split['validation']
# print(len(part_data))

# intervals = list(map(lambda q: [2000 * q, 2000 * (q+1)], list(range(6))))

# for [left, right] in intervals:
#     try:
#         os.mkdir("{0}/ims_val/{1}_{2}".format(DATA_ROOT, left, right))
#         os.mkdir("{0}/anns_val/{1}_{2}".format(DATA_ROOT, left, right))
#     except:
#         pass


for idx, (vehicle_id, label) in enumerate(part_data):
    # for [left, right] in intervals:
    #     if idx in range(left, right):
    #         folder = '{}_{}'.format(left, right)
    
    parse_vehicle_filter(vehicle_id, '')
        


# instance_number = 0
# for idx, (vehicle_id, label) in enumerate(part_data):

#     if idx in range(8000, 10000):
#         vehicle, instance_, bb3d_ = dataset.get_vehicle_instance_data(vehicle_id, 0)
#         instance_number += len(vehicle["instances"])


# print(instance_number)
