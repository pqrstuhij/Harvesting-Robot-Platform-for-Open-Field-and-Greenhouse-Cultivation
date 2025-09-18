import cv2
import numpy
import torch

from typing import Tuple, Union
from enum import Enum

from utils.image_processing import draw_bounding_box, get_common_area, get_min_rect_bounding_box, get_center_of_gravity, get_vertex_order_of_box

HUE_RANGE = [(0, 9), (167, 179)]


class PepperMode(Enum):
    ALL = 0       # 모든 고추 탐색
    GREEN = 1     # 초록고추만 탐색
    RED = 2       # 빨간고추만 탐색


class Pepper:
    def __init__(
        self,
        image: cv2.typing.MatLike,
        pepper_box: torch.Tensor,
        stem_mask: Union[torch.Tensor, numpy.ndarray],
        body_mask: Union[torch.Tensor, numpy.ndarray]
    ):
        """
        고추를 나타내는 클래스입니다.

        Args:
            image (cv2.typing.MatLike): 고추 이미지
            pepper_box (torch.Tensor): 고추 바운딩 박스
            stem_mask (Union[torch.Tensor, numpy.ndarray]): 줄기 마스크
            body_mask (Union[torch.Tensor, numpy.ndarray]): 고추 몸통 마스크

        Attributes:
            crop_shape (Tuple[int, int]): 고추 이미지 크기
            stem_mask (cv2.typing.MatLike): 줄기 마스크
            body_mask (cv2.typing.MatLike): 고추 몸통 마스크
            pepper_box (torch.Tensor): 고추 바운딩 박스
            stem_box (cv2.RotatedRect): 줄기 바운딩 박스
            body_box (cv2.RotatedRect): 고추 몸통 바운딩 박스
            stem_center (Tuple[int, int]): 줄기 중심
            body_center (Tuple[int, int]): 고추 몸통 중심

        """

        # 줄기 마스크 및 고추 몸통 마스크를 생성
        self.stem_mask: cv2.typing.MatLike = stem_mask
        self.body_mask: cv2.typing.MatLike = body_mask

        self.pepper_box = pepper_box
        # 줄기 및 고추 몸통의 바운딩 박스 계산
        self.stem_box: cv2.RotatedRect = get_min_rect_bounding_box(
            self.stem_mask)
        self.body_box: cv2.RotatedRect = get_min_rect_bounding_box(
            self.body_mask)

        # 줄기 및 고추 몸통의 중심 계산
        self.stem_center: Tuple[int, int] = get_center_of_gravity(
            self.stem_mask)
        self.body_center: Tuple[int, int] = get_center_of_gravity(
            self.body_mask)

        # 고추의 윗 부분 계산
        self.top_line: Tuple[Tuple[int, int],
                             Tuple[int, int]] = self.get_top_line()
        # 고추의 윗 부분 중심 계산
        self.top_point: Tuple[int, int] = get_center_of_gravity(
            get_common_area(self.stem_mask, self.top_line))

        self.cutting_line: Tuple[Tuple[int, int],
                                 Tuple[int, int]] = None

        """고추의 성숙도, 자르는 지점, 자르는 각도 계산"""
        """      로봇이 고추를 자르기 위한 정보     """
        self.maturity: float = self.get_maturity(image)
        self.cutting_point: Tuple[int, int] = self.get_cutting_point()
        self.cutting_angle: numpy.ndarray = self.get_cutting_angle()
        """---------------------------------"""

    def get_maturity(self, image: cv2.typing.MatLike) -> float:
        """
        고추의 성숙도를 계산합니다.

        Args:
            image (cv2.typing.MatLike): 고추 이미지

        Returns:
            float: 성숙도
        """
        # 이미지를 HSV 색상 공간으로 변환
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 고추 몸통 마스크를 적용하여 이미지 추출
        body_mask = cv2.bitwise_and(hsv_image, hsv_image, mask=self.body_mask)

        # 빨간색 영역을 추출하기 위한 마스크 생성
        red_mask = numpy.zeros_like(body_mask[:, :, 2], dtype=numpy.uint8)
        for hue_range in HUE_RANGE:
            red_mask = cv2.bitwise_or(
                red_mask, cv2.inRange(body_mask[:, :, 0], hue_range[0], hue_range[1]))
        red_mask = cv2.bitwise_and(red_mask, self.body_mask)

        # 빨간색 영역의 비율 계산
        return numpy.count_nonzero(
            red_mask) / numpy.count_nonzero(self.body_mask) * 100

    def get_top_line(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        고추의 윗 부분을 반환합니다.

        Returns:
            Tuple[Tuple[int, int], Tuple[int, int]]: 고추의 윗 부분
        """
        # 줄기 바운딩 박스의 꼭지점을 기준으로 정렬
        vertex = get_vertex_order_of_box(self.stem_box, self.body_center)

        # vertex 리스트가 충분한 요소를 가지고 있는지 확인
        if len(vertex) < 4:
            # 기본값을 반환하거나 None 처리
            if self.stem_center is not None and self.body_center is not None:
                return ((self.stem_center[0], self.stem_center[1]),
                        (self.body_center[0], self.body_center[1]))
            else:
                return ((0, 0), (0, 0))

        return (int(vertex[2][0]), int(vertex[2][1])), (int(vertex[3][0]), int(vertex[3][1]))

    def get_cutting_point(self) -> Tuple[int, int]:
        """
        고추의 자르는 지점을 반환합니다.

        Returns:
            Tuple[int, int]: 고추의 자르는 지점
        """
        # None 체크를 추가하여 안전하게 처리
        if self.top_point is None or self.stem_center is None:
            return (0, 0) if self.stem_center is None else self.stem_center

        if self.stem_box is None:
            return self.stem_center

        # 고추의 윗 부분의 중심과 줄기 중심 사이의 거리를 계산 후 0.6배 지점을 계산
        x = int(self.top_point[0] + (self.stem_center[0] -
                                     self.top_point[0]) * 0.5)
        y = int(self.top_point[1] + (self.stem_center[1] -
                                     self.top_point[1]) * 0.5)

        # 고추의 줄기 방향의 수직 방향 벡터 계산
        dx = numpy.cos(numpy.radians(self.stem_box[2] + 90))
        dy = numpy.sin(numpy.radians(self.stem_box[2] + 90))

        new_line = ((int(x - 100 * dx), int(y - 100 * dy)),
                    (int(x + 100 * dx), int(y + 100 * dy)))

        # 자르는 지점 계산
        cutting_point = get_center_of_gravity(
            get_common_area(self.stem_mask, new_line))

        # 자르는 지점이 None인 경우, 줄기 중심을 사용
        if cutting_point is None:
            cutting_point = self.stem_center

        # 새로운 방향 벡터 계산
        new_x = self.stem_center[0] - cutting_point[0]
        new_y = self.stem_center[1] - cutting_point[1]
        degree = numpy.arctan2(new_y, new_x)
        new_dx = numpy.cos(degree + numpy.radians(90))
        new_dy = numpy.sin(degree + numpy.radians(90))

        self.cutting_line = ((int(cutting_point[0] - 50 * new_dx), int(cutting_point[1] - 50 * new_dy)),
                             (int(cutting_point[0] + 50 * new_dx), int(cutting_point[1] + 50 * new_dy)))

        return (cutting_point[0], cutting_point[1])

    def get_cutting_angle(self) -> float:
        """
        고추의 자르는 각도를 반환합니다.

        Returns:
            float: 고추의 자르는 각도
        """
        dx = self.cutting_point[0] - self.stem_center[0]
        dy = self.stem_center[1] - self.cutting_point[1]

        angle = numpy.degrees(numpy.arctan2(dy, dx))

        return angle

    def set_maturity(self, maturity: float):
        """
        고추의 성숙도를 설정합니다.

        Args:
            maturity (float): 성숙도
        """
        self.maturity = maturity

    def set_cutting_point(self, cutting_point: Tuple[int, int]):
        """
        고추의 자르는 지점을 설정합니다.

        Args:
            cutting_point (Tuple[int, int]): 자르는 지점
        """
        self.cutting_point = cutting_point

    def set_cutting_angle(self, cutting_angle: float):
        """
        고추의 자르는 각도를 설정합니다.

        Args:
            cutting_angle (float): 자르는 각도
        """
        self.cutting_angle = cutting_angle

    def show_debug_info(self, image: cv2.typing.MatLike, mature_threshold: int = 0, pepper_mode: PepperMode = PepperMode.ALL) -> cv2.typing.MatLike:
        """
        디버그 정보를 표시합니다.

        Args:
            image (cv2.typing.MatLike): 이미지
            mature_threshold (int, optional): 성숙도 기준. Defaults to 0.
            pepper_mode (PepperMode, optional): 고추 모드. Defaults to PepperMode.ALL.

        Returns:
            cv2.typing.MatLike: 디버그 정보가 표시된 이미지
        """
        # 성숙도 기준에 따라 고추를 필터링

        if pepper_mode == PepperMode.GREEN and self.maturity >= mature_threshold:
            return image
        elif pepper_mode == PepperMode.RED and self.maturity < mature_threshold:
            return image

        result = image.copy()

        # 디버그용 줄기 및 고추 몸통 바운딩 박스 그리기
        if self.stem_box is not None:
            debug_stem_box = (
                (self.stem_box[0][0],
                 self.stem_box[0][1]),
                self.stem_box[1],
                self.stem_box[2]
            )
            draw_bounding_box(result, debug_stem_box, (0, 255, 0))

        if self.body_box is not None:
            debug_body_box = (
                (self.body_box[0][0],
                 self.body_box[0][1]),
                self.body_box[1],
                self.body_box[2]
            )
            draw_bounding_box(result, debug_body_box, (0, 0, 255))

        # # 디버그용 줄기 중심 및 고추 몸통 중심 그리기
        # if self.stem_center is not None:
        #     debug_stem_center = (int(self.stem_center[0]),
        #                          int(self.stem_center[1]))
        #     cv2.circle(result, debug_stem_center, 5, (0, 255, 0), -1)

        # if self.body_center is not None:
        #     debug_body_center = (int(self.body_center[0]),
        #                          int(self.body_center[1]))
        #     cv2.circle(result, debug_body_center, 5, (0, 0, 255), -1)

        # # 디버그용 고추의 윗 부분 및 윗 부분 중심 그리기
        # if self.top_line is not None:
        #     cv2.line(result, (int(self.top_line[0][0]), int(self.top_line[0][1])), (int(
        #         self.top_line[1][0]), int(self.top_line[1][1])), (255, 0, 0), 2)

        # if self.top_point is not None:
        #     cv2.circle(result, (int(self.top_point[0]), int(
        #         self.top_point[1])), 5, (255, 0, 0), -1)

        # 디버그용 자르는 지점 및 자르는 각도 그리기
        if self.cutting_point is not None:
            cv2.circle(result, (int(self.cutting_point[0]), int(
                self.cutting_point[1])), 5, (255, 0, 255), -1)

        if self.cutting_line is not None:
            cv2.line(result, (int(self.cutting_line[0][0]), int(self.cutting_line[0][1])), (int(
                self.cutting_line[1][0]), int(self.cutting_line[1][1])), (255, 0, 255), 2)

        """고추의 성숙도, 자르는 지점, 자르는 각도 출력"""
        print(f"성숙도: {self.maturity:.2f}%")
        print(f"자르는 지점: {self.cutting_point}")
        print(f"자르는 각도: {self.cutting_angle:.2f}°")
        """---------------------------------"""

        return result, self.cutting_point, self.cutting_angle