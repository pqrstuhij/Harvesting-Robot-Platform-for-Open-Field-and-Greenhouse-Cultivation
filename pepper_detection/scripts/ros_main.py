#!/usr/bin/env python3

import os
import cv2
import sys
import torch
import rospy
import numpy as np
import numpy.linalg as LA
from datetime import datetime

from typing import List
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from ultralytics import YOLO
from ultralytics.engine.results import Results
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from pepper_detection.msg import PepperInfo 

# 현재 디렉토리 설정 및 모듈 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "src"))

from features.classification import Pepper, PepperMode
from utils.image_processing import crop_image, generate_mask, subtract_mask

# ================== 파라미터 설정 ==================
PM_VERSION = "2.4.32"
SM_VERSION = "1.4.33"

DEBUG = True

VERBOSE = True

SAVE = False  # 이미지 저장을 위한 변수

MATURE_THRESHOLD = 30
CURRENT_MODE = PepperMode.RED
NO_DETECTION_TIMEOUT = 15  # 20초 타임아웃

CALCULATION_FRAME = 5
MAX_COORD_STD = 15
MAX_ANGLE_STD = 10

# ROS 관련 설정
NODE_NAME = "pepper_detection_node"
IMAGE_TOPIC = "/camera/color/image_raw"
OUTPUT_TOPIC = "/pepper_detection/output_image"
PEPPER_INFO_TOPIC = "/pepper_detection/pepper_info"
OUTPUT_IMAGE = "/home/ssenes/catkin_ws/src/senes/pepper_detection/result"

# 모델 경로 설정
MODEL_PATH = os.path.join(current_dir, "..", "models")
# print(MODEL_PATH)

pm_model = YOLO(os.path.join(MODEL_PATH, f"PMv{PM_VERSION}.pt"))
sm_model = YOLO(os.path.join(MODEL_PATH, f"SMv{SM_VERSION}.pt"))

try:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except AssertionError:
    device = torch.device("cpu")

def detect_pepper(image: np.ndarray, device: torch.device) -> List[Pepper]:
    results: List[Pepper] = []
    peppers: Results = pm_model(image, device=device, verbose=False)
    for pepper in peppers:
        output_frame = pepper.plot()
        cv2.imshow("Output", output_frame)

        for p in pepper:
            pepper_mask = generate_mask(image.shape, p.masks.data[0])
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.dilate(pepper_mask, kernel, iterations=5)
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            crop = crop_image(masked_image, p.boxes.xyxy[0], scale=10)

            stems: Results = sm_model(masked_image, device=device, verbose=True)
            best_stem = max(
                (s for stem in stems for s in stem),
                key=lambda s: s.summary()[0]['confidence'],
                default=None
            )

            if best_stem:
                stem_mask = generate_mask(image.shape, best_stem.masks.data[0])
                body_mask = subtract_mask(pepper_mask, stem_mask)
                results.append(Pepper(
                    image=image,
                    pepper_box=p.boxes.xyxy[0],
                    stem_mask=stem_mask,
                    body_mask=body_mask
                ))

    return results

def process_pepper_frames(pepper_frame: List[List[Pepper]], frames: List[cv2.typing.MatLike]) -> List[Pepper]:
    new_pepper: List[Pepper] = []

    if all(len(p) == len(pepper_frame[0]) for p in pepper_frame):
        all_maturity = [[] for _ in pepper_frame[0]]
        all_cutting_points = [[] for _ in pepper_frame[0]]
        all_cutting_angles = [[] for _ in pepper_frame[0]]

        for peppers in pepper_frame:
            for i, pepper in enumerate(peppers):
                all_maturity[i].append(pepper.maturity)
                all_cutting_points[i].append(pepper.cutting_point)
                all_cutting_angles[i].append(pepper.cutting_angle)

        result = frames[0].copy()
        for i, peppers in enumerate(pepper_frame[0]):
            cutting_points = np.array(all_cutting_points[i])
            cutting_angles = np.array(all_cutting_angles[i])
            std_coord = np.std(cutting_points, axis=0)
            coord_dev = LA.norm(std_coord)
            angle_dev = np.std(cutting_angles)

            if coord_dev < MAX_COORD_STD and angle_dev < MAX_ANGLE_STD:
                maturity = np.mean(all_maturity[i])
                cutting_point = np.mean(cutting_points, axis=0)
                cutting_angle = np.mean(cutting_angles)

                peppers.maturity = maturity
                peppers.cutting_point = cutting_point
                peppers.cutting_angle = cutting_angle

                new_pepper.append(peppers)

                if DEBUG:
                    result, point, angle = peppers.show_debug_info(result, MATURE_THRESHOLD)

    return result, point, angle

class PepperDetectionNode:
    def __init__(self):
        rospy.init_node(NODE_NAME, anonymous=True)
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback, queue_size=1)
        self.image_pub = rospy.Publisher(OUTPUT_TOPIC, Image, queue_size=1)
        self.pepper_info_pub = rospy.Publisher(PEPPER_INFO_TOPIC, PepperInfo, queue_size=1)
        rospy.Subscriber("/detection_enabled", Bool, self.detection_enabled_callback)

        self.detection_enabled = True
        self.last_detection_time = datetime.now()
        rospy.loginfo(f"{NODE_NAME} 시작됨. 이미지 토픽: {IMAGE_TOPIC}")

        self.pepper_frame = []
        self.frame_list: List[cv2.typing.MatLike] = []

        self.count = 0

    def detection_enabled_callback(self, msg):
        self.detection_enabled = msg.data
        rospy.loginfo(f"Detection 상태 변경: {'활성화' if self.detection_enabled else '비활성화'}")
        
        # 추가된 부분: detection 활성화될 때마다 시간 초기화
        if self.detection_enabled:
            self.last_detection_time = datetime.now()

    def check_detection_timeout(self):
        # 추가된 부분: detection 비활성화 시 타임아웃 체크 안함
        if not self.detection_enabled:
            return
            
        if (datetime.now() - self.last_detection_time).total_seconds() > NO_DETECTION_TIMEOUT:
            rospy.logwarn(f"{NO_DETECTION_TIMEOUT}초 이상 고추 인식 실패. -1 값 발행")
            
            pepper_info_msg = PepperInfo()
            pepper_info_msg.point = Point(-1, -1, -1)
            pepper_info_msg.angle = -1
            pepper_info_msg.bbox_center = Point(-1, -1, -1)
            self.pepper_info_pub.publish(pepper_info_msg)

            self.last_detection_time = datetime.now()

    def image_callback(self, msg):
        if not self.detection_enabled:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge 변환 실패: {e}")
            return

        peppers = detect_pepper(cv_image, device)

        if SAVE:
            filename = os.path.join(OUTPUT_IMAGE, f"frame_{self.count:04d}.jpg")
            self.count += 1
            cv2.imwrite(filename, cv_image)
            
        if peppers:
            self.last_detection_time = datetime.now()
        else:
            self.check_detection_timeout()

        if DEBUG:
            result_image = cv_image.copy()
            for pepper in peppers:
                result_image, point, angle = pepper.show_debug_info(result_image, MATURE_THRESHOLD)
                
                bbox_center_x = (pepper.pepper_box[0] + pepper.pepper_box[2]) / 2
                bbox_center_y = (pepper.pepper_box[1] + pepper.pepper_box[3]) / 2
                
                pepper_info_msg = PepperInfo()
                pepper_info_msg.point = Point(point[0], point[1], 0)
                pepper_info_msg.angle = angle
                pepper_info_msg.bbox_center = Point(bbox_center_x, bbox_center_y, 0)
                self.pepper_info_pub.publish(pepper_info_msg)
                


            cv2.imshow("Pepper Detection", result_image)
            cv2.waitKey(1)

        try:
            output_msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
            self.image_pub.publish(output_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge 변환 실패: {e}")

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.check_detection_timeout()
            rate.sleep()

if __name__ == "__main__":
    try:
        node = PepperDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
