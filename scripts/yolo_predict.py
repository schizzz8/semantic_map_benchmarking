#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rospkg
from ctypes import *

from IPython import embed


rospack = rospkg.RosPack()
pkg_path = rospack.get_path('semantic_map_benchmarking')


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


lib = CDLL(os.path.join(pkg_path, 'scripts/yolo/darknet/libdarknet.so'), RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict_p
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

detect = lib.network_predict_p
detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network_p
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

network_detect = lib.network_detect
network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]


def read_meta(path):
    class Meta:
        def __init__(self):
            self.classes = None
            self.names = None

    data = Meta()

    with open(path, 'r') as f:
        names = [bytes(n.strip()) for n in f]

    data.classes = len(names)
    data.names = names

    return data


def detect(net, meta, image, thresh=.2, hier_thresh=.2, nms=.45):
    im = load_image(image, 0, 0)

    im_height = im.h
    im_width = im.w

    boxes = make_boxes(net)
    probs = make_probs(net)
    num = num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []

    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                name = str(meta.names[i].decode('utf-8'))

                x_min = max(0., boxes[j].x - 0.5 * boxes[j].w) / float(im_width)
                x_max = min(im_width, boxes[j].x + 0.5 * boxes[j].w) / float(im_width)
                y_min = max(0., boxes[j].y - 0.5 * boxes[j].h) / float(im_height)
                y_max = min(im_height, boxes[j].y + 0.5 * boxes[j].h) / float(im_height)

                bbox = (x_min, x_max, y_min, y_max)
                res.append((name, probs[j][i], bbox))

    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)

    return res


def predict(input_path, model, meta):
    global yolo

    output = {}
    results = detect(model, meta, input_path)

    for pred in results:
        output.update({pred[0] + '_probabilities': pred[1]})
        output.update({pred[0] + '_bounding_boxes': list(pred[2])})

    print(output)
    return output
