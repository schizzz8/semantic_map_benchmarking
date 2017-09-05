#!/usr/bin/env python

from __future__ import print_function

import roslib
roslib.load_manifest('semantic_map_benchmarking')

import sys
import rospy
import cv2
import imp
import itertools
import os
import rospkg
import numpy as np
from copy import deepcopy
from IPython import embed

from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, Joy
from geometry_msgs.msg import PoseWithCovarianceStamped
from semantic_map_extraction.msg import Obs
from cv_bridge import CvBridge, CvBridgeError
import message_filters

import yolo_predict as yolo

desired_classes = ['chair',
                   'diningtable',
                   'table', 'window', 'sofa', 'bookcase', 'wall',
                   'board', 'closet', 'door', 'office', 'ceiling', 'floor'
                   ]
robotname = ''


class YoloDetector:

    def __init__(self, model_path):
        self.model = None
        self.obs_pub = rospy.Publisher(robotname + '/ObservationTopic', Obs, queue_size=10)
        self.asr_pub = rospy.Publisher(robotname + '/ASR', String, queue_size=10)
        self.bridge = CvBridge()
        self.depth_image_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)
        self.rgb_image_sub = message_filters.Subscriber('/camera/rgb/image_raw', Image)
        self.rgb_info_sub = message_filters.Subscriber('/camera/rgb/camera_info', CameraInfo)
        self.amcl_sub = message_filters.Subscriber('/amcl_pose', PoseWithCovarianceStamped)
        self.enable = False
        self.joy_sub = rospy.Subscriber('joy', Joy, self.joyCallback)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.depth_image_sub, self.rgb_image_sub, self.rgb_info_sub, self.amcl_sub], 10, 0.1)
        self.ts.registerCallback(self.imageCallback)
        self.model = yolo.load_net(bytes(os.path.join(model_path, '/scripts/yolo/darknet/cfg/yolo.2.0.cfg')),
                                   bytes(os.path.join(model_path, '/scripts/yolo/yolo.2.0.weights')), 0)
        self.meta = yolo.read_meta(os.path.join(model_path, '/scripts/yolo/darknet/data/coco.names'))

    def joyCallback(self, joy_data):
        if joy_data.buttons[0] == 1:
            rospy.loginfo('capturing image!')
            self.enable = True

    def imageCallback(self, depth_data, rgb_data, rgb_camera_info, pose):
        try:
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding="32FC1")
            cv_rgb_image = self.bridge.imgmsg_to_cv2(rgb_data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        focal_length = rgb_camera_info.K[0]

        if self.enable == True:
            print("Evaluating image")

            cv2.imwrite('/tmp/predict.png', cv_rgb_image)

            if self.model is None or self.meta is None:
                return

            prediction = yolo.predict(bytes('/tmp/predict.png'), self.model, self.meta)
            detections = []

            for i, c in enumerate(desired_classes):
                active = False

                if c + '_probabilities' in prediction:
                    cls = prediction[c + '_probabilities']
                    if cls > 0:
                        active = True
                    else:
                        continue

                if c + '_bounding_boxes' in prediction:
                    bbox = prediction[c + '_bounding_boxes']

                    if active:
                        detections.append((i, bbox, cls))
                elif active:
                    detections.append((i, [], cls))

            q = pose.pose.pose.orientation
            yaw = np.arctan2(2.0 * (q.y * q.z + q.w * q.x), q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z)

            for d in detections:
                output = self.filter(cv_depth_image, d[1], focal_length)
                z_distance, obj_x_size, obj_y_size, obj_depth_size = output

                if z_distance == 0. and obj_depth_size == 0.:
                    continue

                obs = Obs()
                obs.stamp = rospy.Time.now()
                obs.posx = pose.pose.pose.position.x + np.cos(yaw) * z_distance
                obs.posy = pose.pose.pose.position.y + np.sin(yaw) * z_distance
                obs.theta = 0
                obs.dimx = obj_x_size
                obs.dimy = obj_y_size
                obs.dimz = obj_depth_size
                obs.properties = 'color:red'
                self.obs_pub.publish(obs)
                print('Class', desired_classes[d[0]], 'probability', d[2])
                string = '("' + desired_classes[d[0]] + '")'
                self.asr_pub.publish(string)

                rospy.sleep(0.1)
            self.enable = False

    def filter(self, image, result, focal_length, camera_height=0.287):
        print('Filtering')
        result = np.atleast_2d(result)
        if result.shape[1] != 4:
            return None

        im_height = image.shape[0]
        im_width = image.shape[1]

        int_bbox = [int(result[0, 2] * im_height),
                    int(result[0, 3] * im_height),
                    int(result[0, 0] * im_width),
                    int(result[0, 1] * im_width)]

        roi = np.atleast_2d(image[int_bbox[0]:int_bbox[1], int_bbox[2]:int_bbox[3]].copy())
        print('roi image size:', str(roi.shape))

        if roi.shape[0] == 0 or roi.shape[1] == 0:
            return 0., 0., 0., 0.

        roi_height = roi.shape[0]
        roi_width = roi.shape[1]
        original_roi = roi.copy()

        image_pred = image.copy()
        image_pred = cv2.rectangle(image_pred, (int_bbox[2], int_bbox[0]), (int_bbox[3], int_bbox[1]), 255)
        cv2.imwrite('/tmp/detection.png', image_pred)
        cv2.imwrite('/tmp/roi.png', original_roi)
        above_horizon_roi = np.inf * np.ones((roi_height, roi_width), np.float32)

        for h in range(0, roi_height):
            for w in range(0, roi_width):
                if (0.5 * im_height - (int_bbox[0] + h)) * original_roi[h, w] / focal_length > -camera_height:
                    above_horizon_roi[h, w] = original_roi[h, w]

        cv2.imwrite('/tmp/above_horizon_roi.png', above_horizon_roi)
        min_pt = np.argmin(above_horizon_roi)
        min_pt_coords = (min_pt % original_roi.shape[1], int(min_pt / original_roi.shape[1]))

        mask = np.zeros((roi_height + 2, roi_width + 2), np.uint8)
        center_pt = (int(0.5 * roi_width), int(0.5 * roi_height))
        interest_pt = (int(0.5 * (min_pt_coords[0] + center_pt[0])), int(0.5 * (min_pt_coords[1] + center_pt[1])))
        # cv2.floodFill(above_horizon_roi, mask, (int(interest_pt[0]), int(
        #     interest_pt[1])), 255, loDiff=0.025, upDiff=0.025)
        cv2.floodFill(above_horizon_roi, mask, min_pt_coords, 255, loDiff=0.025, upDiff=0.025)

        valid_pts = np.where(above_horizon_roi == 255)

        if len(valid_pts) == 0:
            valid_pts = np.zeros((2, 1))

        max_obj_depth = np.amax(original_roi[valid_pts])
        min_obj_depth = np.amin(original_roi[valid_pts])
        min_size = np.amin(valid_pts, axis=1)
        max_size = np.amax(valid_pts, axis=1)

        im_size = max_size - min_size
        obj_y_size = (im_size[0]) * min_obj_depth / focal_length
        obj_x_size = (im_size[1]) * min_obj_depth / focal_length
        obj_depth_size = max_obj_depth - min_obj_depth
        print('y_size:', obj_y_size)
        print('x_size:', obj_x_size)
        print('z_size:', obj_depth_size)

        cv2.imwrite('/tmp/roi_after.png', above_horizon_roi)

        return min_obj_depth, obj_x_size, obj_y_size, obj_depth_size


def main(args):
    rospy.init_node('yolo_detector_node', anonymous=True)
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('semantic_map_benchmarking')
    
    YoloDetector(model_path=pkg_path)
    rospy.loginfo("Starting yolo_detector node")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
