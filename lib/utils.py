# -*- coding: utf-8 -*-
import pickle
import os
import numpy as np
import sys
from math import floor

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

def get_true_angle(bb3d):
    angles = cross_from_points(bb3d)
    for i in range(len(angles)):
        angles[i] = floor(- angles[i] / 3.0) + 30

    return angles

