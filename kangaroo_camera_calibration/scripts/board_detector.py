#!/usr/bin/env python
"""Live ChArUco detection for teaching: annotated image, corner count and board TF.

Publishes (private):
  ~image        annotated camera image (view it in RViz)
  ~num_corners  std_msgs/Int32, detected ChArUco corners (0 if none)
  TF            optical_frame -> charuco_board_detected, while the board is detected
"""

import cv2
import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Int32

from kangaroo_camera_calibration import config, geometry
from kangaroo_camera_calibration.board import CharucoBoard


class Detector(object):
    def __init__(self):
        cfg = config.load()
        self.board = CharucoBoard.from_params(cfg["board"])
        camera = cfg["camera"]
        self.max_rate = rospy.get_param("~max_rate", 5.0)
        self.bridge = CvBridge()
        self.last_processed = rospy.Time(0)
        info = rospy.wait_for_message(camera["info_topic"], CameraInfo, timeout=30.0)
        self.K = np.array(info.K).reshape(3, 3)
        self.D = np.array(info.D)
        self.image_pub = rospy.Publisher("~image", Image, queue_size=1)
        self.count_pub = rospy.Publisher("~num_corners", Int32, queue_size=1)
        self.tf_pub = tf2_ros.TransformBroadcaster()
        rospy.Subscriber(camera["image_topic"], Image, self.on_image, queue_size=1, buff_size=2 ** 24)

    def on_image(self, message):
        if (message.header.stamp - self.last_processed).to_sec() < 1.0 / self.max_rate:
            return
        self.last_processed = message.header.stamp
        image = self.bridge.imgmsg_to_cv2(message, "bgr8")
        detection = self.board.detect(image, self.K, self.D)
        valid = self.board.is_valid(detection)
        count = len(detection.ids) if detection is not None else 0
        self.count_pub.publish(Int32(count if valid else 0))
        if self.image_pub.get_num_connections():
            self.board.draw(image, detection, self.K, self.D)
            text = "corners: {}".format(count)
            if valid:
                text += "  pnp rms: {:.2f}px".format(detection.reprojection_rms)
            color = (0, 200, 0) if valid else (0, 0, 255)
            cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            out = self.bridge.cv2_to_imgmsg(image, "bgr8")
            out.header = message.header
            self.image_pub.publish(out)
        if valid:
            xyz, quat = geometry.to_xyz_quat(geometry.from_rvec_tvec(detection.rvec, detection.tvec))
            t = TransformStamped()
            t.header = message.header
            t.child_frame_id = "charuco_board_detected"
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
            (t.transform.rotation.x, t.transform.rotation.y,
             t.transform.rotation.z, t.transform.rotation.w) = quat
            self.tf_pub.sendTransform(t)


if __name__ == "__main__":
    rospy.init_node("charuco_detector")
    Detector()
    rospy.spin()
