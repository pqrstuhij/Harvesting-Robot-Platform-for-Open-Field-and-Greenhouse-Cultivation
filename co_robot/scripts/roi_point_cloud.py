#!/usr/bin/env python3
import rospy
import tf
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, Bool
from geometry_msgs.msg import Point
import numpy as np
from co_robot.srv import vision_robot
from pepper_detection.msg import PepperInfo

class PointCloudProcessor:
    def __init__(self):
        rospy.init_node('pointcloud_processor')

        self.tf_listener = tf.TransformListener()
        self.cloud_sub = rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.cloud_cb)
        self.pepper_sub = rospy.Subscriber('/pepper_detection/pepper_info', PepperInfo, self.pepper_cb)
        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=10)
        self.enable_pub = rospy.Publisher("/detection_enabled", Bool, queue_size=1)
        self.service_client = rospy.ServiceProxy("/service", vision_robot)

        self.processing = False
        self.current_pepper = None
        self.processing_timeout = rospy.Duration(5.0)
        self.last_processing_time = None
        self.timeout_processing = False

        self.home_positions = {
            "visd3": {"x": 399.733, "y": 0.000, "z": 213.767},
            "vist3": {"x": 326.123, "y": 0.000, "z": 506.785}
        }
        rospy.loginfo("PointCloudProcessor initialized")

    def re_enable_detection(self):
        self.enable_pub.publish(Bool(data=True))
        self.processing = False
        self.timeout_processing = False
        self.current_pepper = None
        self.last_processing_time = None
        rospy.loginfo("Detection re-enabled (state reset)")

    def pepper_cb(self, msg):
        if msg.point.x == -1 and msg.point.y == -1 and msg.angle == -1:
            if not self.timeout_processing:
                self.timeout_processing = True
                self.processing = True
                self.enable_pub.publish(Bool(data=False))
                try:
                    rospy.loginfo("No pepper detected for 20s. Calling service (9,0,0,0,0,0)...")
                    response = self.service_client(9, 0, 0, 0, 0, 0, 0)
                    rospy.loginfo(f"Service response: {response.state}")
                except rospy.ServiceException as e:
                    rospy.logerr(f"Service call failed: {e}")
                finally:
                    self.re_enable_detection()
            return

        if self.processing and not self.timeout_processing:
            if self.last_processing_time and (rospy.Time.now() - self.last_processing_time > self.processing_timeout):
                rospy.logwarn("Processing timeout detected. Resetting state.")
                self.re_enable_detection()
            return

        self.current_pepper = {
            "x": int(msg.point.x),
            "y": int(msg.point.y),
            "angle": float(msg.angle),
            "bbox_x": int(msg.bbox_center.x),
            "bbox_y": int(msg.bbox_center.y)
        }

        self.enable_pub.publish(Bool(data=False))
        self.processing = True
        self.last_processing_time = rospy.Time.now()
        rospy.loginfo(f"Pepper detected at ({self.current_pepper['x']}, {self.current_pepper['y']})")

    
    def filter_pointcloud(self, msg, min_z=0.2, max_z=0.5):
        """
        PointCloud2 msg를 받아 z값(min_z~max_z) 범위만 남긴 새 메시지 반환
        """
        filtered_points = []
        for p in pc2.read_points(msg, skip_nans=True):
            if min_z <= p[2] <= max_z:
                filtered_points.append(p)
        
        filtered_msg = pc2.create_cloud(msg.header, msg.fields, filtered_points)
        return filtered_msg
    
    
    
    def cloud_cb(self, msg):
        
        filtered_msg = self.filter_pointcloud(msg, min_z=0.2, max_z=0.5)

        if not self.processing or not self.current_pepper or self.timeout_processing:
            return
        
        if rospy.Time.now() - self.last_processing_time > self.processing_timeout:
            rospy.logwarn("Processing timeout. Resetting.")
            self.re_enable_detection()
            return
    

        #if not self.processing or not self.current_pepper or self.timeout_processing:
        #    return

        #if rospy.Time.now() - self.last_processing_time > self.processing_timeout:
        #    rospy.logwarn("Processing timeout. Resetting.")
        #    self.re_enable_detection()
        #    return

        try:
            (trans, rot) = self.tf_listener.lookupTransform('/base', '/camera_link1', rospy.Time(0))
            tf_matrix = tf.transformations.quaternion_matrix(rot)
            tf_matrix[:3, 3] = trans

            state = rospy.get_param("/current_state", "vist3")
            home = self.home_positions.get(state, self.home_positions["vist3"])
            grip_offset = 160  # mm

            stem_point = self.get_closest_valid_point(
                self.current_pepper["x"],
                self.current_pepper["y"],
                msg,
                tf_matrix,
                window=0
            )

            if stem_point is None:
                rospy.logwarn("Stem point extraction failed. Retrying with wider window...")
                self.re_enable_detection()
                return
 
               

            x1 = (stem_point[0] * 1000) - home["x"] - grip_offset
            y1 = (stem_point[1] * 1000) - home["y"]
            z1 = (stem_point[2] * 1000) - home["z"]

            if state == "vist3":
                if x1 > 350:
                    rospy.logwarn("stem x1 > 250 in vist3, retrying stem extraction...")
                    self.re_enable_detection()
                    return
                    
                    
            #elif state == "visd3":
            #    if x1 > 270:
            #        rospy.logwarn("stem x1 > 180 in visd3, retrying stem extraction...")
            #        self.re_enable_detection()
            #        return
                    
                    

            body_point = self.get_closest_valid_point(
                self.current_pepper["bbox_x"],
                self.current_pepper["bbox_y"],
                msg,
                tf_matrix,
                window=0
            )

            if body_point is None:
                rospy.logwarn("Stem point extraction failed. Retrying with wider window...")
                self.re_enable_detection()
                return
 
                
                
            x2 = (body_point[0] * 1000) - home["x"] - grip_offset
            y2 = (body_point[1] * 1000) - home["y"]
            z2 = (body_point[2] * 1000) - home["z"]

            try:
                resp = self.service_client(9, 1, 0, x1, y1, z1, self.current_pepper["angle"])
                if resp.state == 100:
                    rospy.loginfo(f"[Service Call Success] x1={x1:.1f}, y1={y1:.1f}, z1={z1:.1f}, angle={self.current_pepper['angle']}")
                    marker = Marker()
                    marker.header.frame_id = "base"
                    marker.header.stamp = rospy.Time.now()
                    marker.type = Marker.SPHERE
                    marker.pose.position = Point(
                        #(x2 + home["x"] + grip_offset) / 1000.0,
                        (x1 + home["x"] + grip_offset) / 1000.0,
                        (y1 + home["y"]+9) / 1000.0,
                        (z1 + home["z"]+10) / 1000.0
                    )
                    marker.scale.x = marker.scale.y = marker.scale.z = 0.01
                    marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
                    self.marker_pub.publish(marker)
                else:
                    rospy.logwarn(f"Service failed: State code {resp.state}")
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")

            self.re_enable_detection()

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF error: {e}")
            self.re_enable_detection()
        except Exception as e:
            rospy.logerr(f"Point cloud processing error: {e}")
            self.re_enable_detection()

    def get_closest_valid_point(self, center_x, center_y, depth_msg, tf_matrix, window=None, retry=False):
        min_z = float('inf')
        best_point = None
        window_size = window if window is not None else (7 if retry else 5)

        for dx in range(-window_size, window_size + 1):
            for dy in range(-window_size, window_size + 1):
                u = center_x + dx
                v = center_y + dy
                gen = pc2.read_points(depth_msg, field_names=("x", "y", "z"), skip_nans=True, uvs=[[u, v]])
                points = list(gen)
                if not points:
                    continue
                p = points[0]
                if 0.2 < p[2] < 1.5:
                    if p[2] < min_z:
                        min_z = p[2]
                        best_point = p

        if best_point is None:
            return None

        camera_point = np.array([best_point[0], best_point[1], best_point[2], 1.0])
        world_point = np.dot(tf_matrix, camera_point)

        return world_point[:3]

if __name__ == '__main__':
    try:
        processor = PointCloudProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated")

