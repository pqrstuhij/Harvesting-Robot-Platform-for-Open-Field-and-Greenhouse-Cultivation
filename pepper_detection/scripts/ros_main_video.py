#!/usr/bin/env python3

import os
import cv2
import sys
import torch
import rospy
import numpy as np
import numpy.linalg as LA

from typing import List
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from ultralytics import YOLO
from ultralytics.engine.results import Results
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from pepper_detection.msg import PepperInfo

# 현재 디렉토리 설정 및 모듈 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "src"))

from features.classification import Pepper
from utils.image_processing import crop_image, generate_mask, subtract_mask

# ================== 파라미터 설정 ==================
PM_VERSION = "2.4.21"
SM_VERSION = "1.4.22"

DEBUG = True
MATURE_THRESHOLD = 70

CALCULATION_FRAME = 5
MAX_COORD_STD = 15  # 좌표 안정성 필터링 기준 (픽셀)
MAX_ANGLE_STD = 10  # 각도 안정성 필터링 기준 (degrees)

# ROS 관련 설정
NODE_NAME = "pepper_detection_node"
IMAGE_TOPIC = "/video_frames"
OUTPUT_TOPIC = "/pepper_detection/output_image"
PEPPER_INFO_TOPIC = "/pepper_detection/pepper_info"

# 모델 경로 설정
MODEL_PATH = os.path.join(current_dir, "..", "models")
print(MODEL_PATH)

# 모델 로드
pm_model = YOLO(os.path.join(MODEL_PATH, f"PMv{PM_VERSION}.pt"))
sm_model = YOLO(os.path.join(MODEL_PATH, f"SMv{SM_VERSION}.pt"))

# CUDA 또는 CPU 사용 여부 확인
try:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except AssertionError:
    device = torch.device("cpu")

def detect_pepper(image: np.ndarray, device: torch.device) -> List[Pepper]:
    detected_peppers: List[Pepper] = []
    
    # 고추 감지 모델 실행
    peppers: Results = pm_model(image, device=device, verbose=False)
    for pepper in peppers:
        output_frame = pepper.plot()
        cv2.imshow("Output", output_frame)
        for p in pepper:
            # 고추 마스크 생성
            pepper_mask = generate_mask(image.shape, p.masks.data[0])
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.dilate(pepper_mask, kernel, iterations=5)
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            crop = crop_image(masked_image, p.boxes.xyxy[0], scale=10)
            pepper_mask = crop_image(pepper_mask, p.boxes.xyxy[0], scale=10)

            # 줄기 감지 모델 실행
            stems: Results = sm_model(
                crop, device=device, conf=0.6, verbose=False)
            best_stem = max(
                (s for stem in stems for s in stem),
                key=lambda s: s.summary()[0]['confidence'],
                default=None
            )

            if best_stem:
                stem_mask = generate_mask(crop.shape, best_stem.masks.data[0])
                body_mask = subtract_mask(pepper_mask, stem_mask)

                # Pepper 객체 생성 및 결과 리스트에 추가
                detected_peppers.append(Pepper(
                    image=crop,
                    pepper_box=p.boxes.xyxy[0],
                    stem_mask=stem_mask,
                    body_mask=body_mask
                ))

    return detected_peppers

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
            std_dev = np.std(cutting_points, axis=0)
            dist_std = LA.norm(std_dev)

            if dist_std < MAX_COORD_STD:
                maturity = np.mean(all_maturity[i])
                cutting_point = np.mean(all_cutting_points[i], axis=0)
                cutting_angle = np.mean(all_cutting_angles[i])

                peppers.maturity = maturity
                peppers.cutting_point = cutting_point
                peppers.cutting_angle = cutting_angle

                new_pepper.append(peppers)

                if DEBUG:
                    result = peppers.show_debug_info(result, MATURE_THRESHOLD)
            else:
                if DEBUG:
                    print(f"🚫 필터링됨: ID={i}, 좌표 std={dist_std:.2f} > {MAX_COORD_STD}")

    return new_pepper

class PepperDetectionNode:
    def __init__(self):
        rospy.init_node(NODE_NAME, anonymous=True)
        self.bridge = CvBridge()
        
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback, queue_size=1)
        self.image_pub = rospy.Publisher(OUTPUT_TOPIC, Image, queue_size=1)
        self.pepper_info_pub = rospy.Publisher(PEPPER_INFO_TOPIC, PepperInfo, queue_size=1)
        
        rospy.loginfo(f"{NODE_NAME} 시작됨. 이미지 토픽: {IMAGE_TOPIC}")
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge 변환 실패: {e}")
            return
        
        peppers = detect_pepper(cv_image, device)
        
        if DEBUG:
            result_image = cv_image.copy()
            for pepper in peppers:
                result_image, point, angle = pepper.show_debug_info(result_image)
                pepper_info_msg = PepperInfo()
                pepper_info_msg.point = Point(point[0], point[1], 0)
                pepper_info_msg.angle = angle
                self.pepper_info_pub.publish(pepper_info_msg)
            
            cv2.imshow("Pepper Detection", result_image)
            cv2.waitKey(1)
        
        try:
            output_msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
            self.image_pub.publish(output_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge 변환 실패: {e}")
    
    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        node = PepperDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
