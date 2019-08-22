# -*- coding: utf-8 -*-
import pickle
import os
import numpy as np
import sys
import cv2
from math import floor, cos, sin, pi, sqrt

angle_numbers = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX

#%%
def load_cache(path, encoding="latin-1", fix_imports=True):
    """
    encoding latin-1 is default for Python2 compatibility
    """
    with open(path, "rb") as f:
        return pickle.load(f, encoding=encoding, fix_imports=True)

#%%
def save_cache(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

#%%
def ensure_dir(d):
    if len(d)  == 0: # for empty dirs (for compatibility with os.path.dirname("xxx.yy"))
        return
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except OSError as e:
            if e.errno != 17: # FILE EXISTS
                raise e

#%%
def parse_args(available_nets):
    import argparse
    default_cache = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "cache"))
    parser = argparse.ArgumentParser(description="BoxCars fine-grained recognition algorithm Keras re-implementation",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--eval", type=str, default=None, help="path to model file to be evaluated")
    parser.add_argument("--resume", type=str, default=None, help="path to model file to be resumed")
    parser.add_argument("--train-net", type=str, default=available_nets[0], help="train on one of following nets: %s"%(str(available_nets)))
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0025, help="learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="run for epochs")
    parser.add_argument("--cache", type=str, default=default_cache, help="where to store training meta-data and final model")
    parser.add_argument("--estimated-3DBB", type=str, default=None, help="use estimated 3DBBs from specified path")
    
    
    args = parser.parse_args()
    assert args.eval is None or args.resume is None, "--eval and --resume are mutually exclusive"
    if args.eval is None and args.resume is None:
        assert args.train_net in available_nets, "--train-net must be one of %s"%(str(available_nets))

    return args

 
#%%
def download_report_hook(block_num, block_size, total_size):
    downloaded = block_num*block_size
    percents = downloaded / total_size * 100
    show_str = " %.1f%%"%(percents)
    sys.stdout.write(show_str + len(show_str)*"\b")
    sys.stdout.flush()
    if downloaded >= total_size:
        print()


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


def get_length_from_points(p1, p2):
    return sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        
def cross_from_points(bb3d, img=None):
    front_lines = [(0, 5, 1, 4), (1, 6, 2, 5), (0, 2, 1, 3)]
    back_lines = [(3, 6, 2, 7), (3, 4, 0, 7), (4, 6, 5, 7)]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    angles = []
    for f, b, color in zip(front_lines, back_lines, colors):
        L1 = line(bb3d[f[0]], bb3d[f[1]])
        L2 = line(bb3d[f[2]], bb3d[f[3]])
        R12 = intersection(L1, L2)

        L3 = line(bb3d[b[0]], bb3d[b[1]])
        L4 = line(bb3d[b[2]], bb3d[b[3]])
        R34 = intersection(L3, L4)

        if img is not None:
            cv2.line(img, R12, R34, color, 1)
        angles.append(get_angle_from_two_points(R12, R34))

    if img is not None:
        return img, angles
    else:
        return angles

def three_normalized_dimensions(bb3d, normal_length = 1):
    """
    return dimensions in this order: length, width, height
    """
    dimensions = []
    front_lines = [(0, 5, 1, 4), (1, 6, 2, 5), (0, 2, 1, 3)]
    back_lines = [(3, 6, 2, 7), (3, 4, 0, 7), (4, 6, 5, 7)]
    for f, b in zip(front_lines, back_lines):
        L1 = line(bb3d[f[0]], bb3d[f[1]])
        L2 = line(bb3d[f[2]], bb3d[f[3]])
        R12 = intersection(L1, L2)

        L3 = line(bb3d[b[0]], bb3d[b[1]])
        L4 = line(bb3d[b[2]], bb3d[b[3]])
        R34 = intersection(L3, L4)
        norm_len = get_length_from_points(R12, R34) / normal_length
        dimensions.append(floor(norm_len / 0.015))
        #dimensions.append(norm_len)
    
    return dimensions

def get_true_angle(bb3d):
    angles = cross_from_points(bb3d)
    for i in range(len(angles)):
        angles[i] = floor(- angles[i] / 3.0) + 30

    return angles


def from_angle_to_coordinates(angle, img_size):
    y = img_size[0] // 2
    x = img_size[1] // 2
    if arrow:
        left_point = (x, y)
    else:
        left_point = (round(x - 60 * cos( - angle * pi / 180)), round(y - 60 * sin( - angle * pi / 180)))
    right_point = (round(x + 60 * cos( - angle * pi / 180)), round(y + 60 * sin( - angle * pi / 180)))
    return left_point, right_point


def visualize_prediction_arrows(prediction_per_image, image, image_name, bb3d, angle_idx):
    directions_predictions = prediction_per_image['output_d']
    direction = directions_predictions.index(max(directions_predictions))
    
    if angle_idx < angle_numbers:
        angle = get_angle_from_prediction(prediction_per_image, angle_idx)
        true_angle = - cross_from_points(bb3d)[angle_idx]
        L, R = from_angle_to_coordinates(angle, image.shape)
        true_L, true_R = from_angle_to_coordinates(true_angle, image.shape)

    else:
        L_R = []
        for i in range(angle_numbers):
            angle = get_angle_from_prediction(prediction_per_image, 2-i)
            L, R = from_angle_to_coordinates(angle, image.shape)
            L_R.append((L, R))

    if direction == 1:
        text = 'to_camera'
        color = (0,0,255)
    else:
        text = 'from_camera'
        color = (0,255,0)

    if angle_idx < angle_numbers:
        cv2.arrowedLine(image, true_L, true_R, (255, 0, 0), 3)
        cv2.putText(image, 'true_angle: {}'.format(true_angle), (0, 40), FONT, 0.5, (255,0,0), 1, cv2.LINE_AA)

        cv2.arrowedLine(image, L, R, color, 3)	# need one per angle
        cv2.putText(image, 'predict_angle{}: {}'.format(angle_idx, angle), (0, 20), FONT, 0.5, color, 1, cv2.LINE_AA)  # print predicted angles.need one per angle

    else:
        colors = [(0,0,0), (0,255,255), (255, 0, 0)]
        try:
            coordinate_color = list(zip(L_R, colors))
        except AssertionError:
            print('more than 3 angles')
        
        for i, c_and_c in enumerate(coordinate_color):
            (L, R), arrow_color = c_and_c
            cv2.arrowedLine(image, L, R, arrow_color, 3)	# need one per angle            
            cv2.putText(image, 'predict_angle{}: {}'.format(i, angle), (0, 15 * (i + 1)), FONT, 0.5, arrow_color, 1, cv2.LINE_AA)  # print predicted angles.need one per angle

    cv2.putText(image, text, (0, image.shape[0]), FONT, 1, color, 1, cv2.LINE_AA) #print direction, need one line in a image anyways

    cv2.imwrite(image_name, image)

def get_angle_from_prediction(prediction_per_image, angle_idx):
    key = 'output_a{}'.format(angle_idx)
    prediction = prediction_per_image[key]
    angle = (prediction.index(max(prediction)) - 30) * 3.0

    return angle


def get_point_from_angle(angle):
    return np.array([round(20 * cos( - angle * pi / 180)), round(20 * sin( - angle * pi / 180))])

def visualize_prediction_boxes(prediction_per_image, image, cropped_img = False, crop_coordinates = np.zeros([4], dtype='int32')):
    if not cropped_img:
        crop_coordinates[2] = image.shape[1]
        crop_coordinates[3] = image.shape[0] # coordinate order: x, y, width, height

    directions_predictions = prediction_per_image['output_d']
    direction = directions_predictions.index(max(directions_predictions))

    points = np.zeros([angle_numbers, 2], dtype='int32')
    for i in range(angle_numbers):
        angle = get_angle_from_prediction(prediction_per_image, i)
        points[i] = get_point_from_angle(angle)
       
    y = crop_coordinates[1] + crop_coordinates[3] // 2
    x = crop_coordinates[0] + crop_coordinates[2] // 2
    base_point = np.array([x, y], dtype = 'int32')

    img_scale_x = crop_coordinates[2] / 110
    img_scale_y = crop_coordinates[3] / 110
    scale = np.array([img_scale_x, img_scale_y], dtype='float32')

    coordinates = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                raw_coordinate = - (k*j) * points[0] - j * points[1] - i * points[2]
                coordinate = np.multiply(scale, raw_coordinate, casting='unsafe', dtype='float32') + base_point
                int_coordinate = coordinate.astype(dtype = 'int64', casting = 'unsafe')
                coordinates.append(tuple(int_coordinate))
    
    for u in range(4):
        cv2.line(image, coordinates[u], coordinates[(u + 1) % 4], (255, 0, 0), 2)	# need one per angle            
        cv2.line(image, coordinates[u], coordinates[u + 4], (0, 0, 0), 2)	# need one per angle            
        cv2.line(image, coordinates[u + 4], coordinates[(u + 1)%4 + 4], (255, 0, 0), 2)	# need one per angle            
    
    if direction == 1:
        text = 'to_camera'
        color = (0,0,255)
    else:
        text = 'from_camera'
        color = (0,255,0)
    cv2.putText(image, text ,(crop_coordinates[0], crop_coordinates[1] + crop_coordinates[3]), FONT, 1, color, 1, cv2.LINE_AA) #print direction, need one line in a image anyways

    return image


def image_preprocess(im):
    
    desired_size = 224

    old_size = im.shape[:2]  # im.shape is in (height, width, channel) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = cv2.resize(im, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)

    # create a new image and paste the resized on it
    new_im = np.zeros((desired_size, desired_size, 3), dtype=np.uint8)
    offset_x = (desired_size-new_size[1])//2
    offset_y = (desired_size-new_size[0])//2
    new_im[offset_y: offset_y + new_size[0], offset_x: offset_x + new_size[1]] = im
    
    return new_im