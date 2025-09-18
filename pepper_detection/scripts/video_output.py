#!/usr/bin/env python
# ROS1 (Python2 or Python3 depending on Noetic build)

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def publish_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        rospy.logerr("Failed to open video file: %s", video_path)
        return

    pub = rospy.Publisher('/video_frames', Image, queue_size=10)
    bridge = CvBridge()
    rospy.init_node('video_publisher_node', anonymous=True)
    rate = rospy.Rate(30)  # Adjust depending on video FPS

    while not rospy.is_shutdown() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            rospy.loginfo("End of video file.")
            break
        msg = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        pub.publish(msg)
        rate.sleep()

    cap.release()

if __name__ == '__main__':
    try:
        publish_video('/home/airo/catkin_ws/src/pepper_detection/data/test_video/new1.mp4')
    except rospy.ROSInterruptException:
        pass
