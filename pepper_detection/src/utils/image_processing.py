import cv2
import numpy
import torch

from typing import List, Tuple, Union


def crop_image(image: cv2.typing.MatLike, box: Union[torch.Tensor, numpy.ndarray], scale: int = 0) -> cv2.typing.MatLike:
    """
    이미지에서 박스 좌표에 해당하는 부분을 잘라냅니다.

    Args:
        image (cv2.typing.MatLike): 이미지
        box (Union[torch.Tensor, numpy.ndarray]): 박스 좌표
        scale (int, optional): 이미지 확대 비율. Defaults to 0.

    Returns:
        cv2.typing.MatLike: 잘라낸 이미지
    """
    x1, y1, x2, y2 = map(int, box)

    # 좌표에 스케일을 적용하여 이미지 경계를 벗어나지 않도록 조정합니다.
    x1 = max(0, x1 - scale)
    y1 = max(0, y1 - scale)
    x2 = min(image.shape[1], x2 + scale)
    y2 = min(image.shape[0], y2 + scale)

    # 이미지를 잘라냅니다.
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image


def generate_mask(image_shape: Tuple[int, int], mask: Union[torch.Tensor, numpy.ndarray]) -> cv2.typing.MatLike:
    """
    마스크를 생성합니다.

    Args:
        image_shape (Tuple[int, int]): 이미지 크기
        mask (Union[torch.Tensor, numpy.ndarray]): 마스크

    Returns:
        cv2.typing.MatLike: 생성된 마스크
    """
    # 마스크를 numpy 배열로 변환하고 크기를 조정합니다.
    mask = (mask.cpu().numpy() * 255).astype(numpy.uint8)
    mask = cv2.resize(mask, (image_shape[1], image_shape[0]))

    return mask


def apply_mask_to_image(image: cv2.typing.MatLike, mask: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """
    이미지에 마스크를 적용합니다.

    Args:
        image (cv2.typing.MatLike): 이미지
        mask (cv2.typing.MatLike): 마스크

    Returns:
        cv2.typing.MatLike: 마스크가 적용된 이미지
    """
    copied_image = image.copy()
    masked_image = cv2.bitwise_and(copied_image, copied_image, mask=mask)
    return masked_image


def subtract_mask(mask1: cv2.typing.MatLike, mask2: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """
    두 마스크를 뺀 결과를 반환합니다.

    Args:
        mask1 (cv2.typing.MatLike): 첫 번째 마스크
        mask2 (cv2.typing.MatLike): 두 번째 마스크

    Returns:
        cv2.typing.MatLike: 두 마스크를 뺀 결과
    """
    subtracted_mask = cv2.subtract(mask1, mask2)
    return subtracted_mask


def get_min_rect_bounding_box(mask: cv2.typing.MatLike) -> Union[cv2.RotatedRect, None]:
    """
    가장 큰 마스크의 바운딩 박스를 추출합니다.

    Args:
        mask (cv2.typing.MatLike): 마스크 이미지

    Returns:
        Union[cv2.RotatedRect, None]: 가장 큰 바운딩 박스 또는 None (윤곽선이 없는 경우)
    """
    # 마스크 이미지에서 외곽선을 찾습니다.
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    # 가장 큰 외곽선을 찾습니다.
    largest_contour = max(contours, key=cv2.contourArea)
    # 가장 큰 외곽선의 최소 영역 바운딩 박스를 계산합니다.
    bounding_box = cv2.minAreaRect(largest_contour)

    return bounding_box


def draw_bounding_box(image: cv2.typing.MatLike, box: cv2.RotatedRect, color: Tuple[int, int, int] = (255, 0, 0)) -> None:
    """
    이미지에 바운딩 박스를 그립니다.

    Args:
        image (cv2.typing.MatLike): 이미지
        box (cv2.RotatedRect): 바운딩 박스
        color (Tuple[int, int, int], optional): 바운딩 박스 색상. Defaults to (255, 0, 0).
    """
    # # 입력값 검증
    # if box is None:
    #     return

    # 바운딩 박스의 꼭지점을 계산합니다.
    box_points = cv2.boxPoints(box)
    box_points = numpy.int32(box_points)
    # 이미지를 따라 바운딩 박스를 그립니다.
    cv2.drawContours(image, [box_points], 0, color, 2)


def get_vertex_order_of_box(box: cv2.RotatedRect, point: Tuple[int, int]) -> List[Tuple[float, float]]:
    """
    바운딩 박스의 꼭지점을 기준점을 기준으로 정렬합니다.

    Args:
        box (cv2.RotatedRect): 바운딩 박스
        point (Tuple[int, int]): 기준점

    Returns:
        List[Tuple[float, float]]: 정렬된 꼭지점 리스트
    """
    # 입력값 검증
    if box is None:
        return []

    if point is None:
        return []

    # 바운딩 박스의 꼭지점을 계산합니다.
    box_points = cv2.boxPoints(box)
    # 각 꼭지점과 기준점 사이의 거리를 계산합니다.
    distances = [numpy.linalg.norm(p - numpy.array(point)) for p in box_points]
    # 거리에 따라 꼭지점을 정렬합니다.
    vertex_order = [vertex for _, vertex in sorted(
        zip(distances, box_points), key=lambda x: x[0])]

    return vertex_order


def get_center_of_gravity(mask: cv2.typing.MatLike) -> Union[Tuple[int, int], None]:
    """
    마스크의 무게중심을 계산합니다.

    Args:
        mask (cv2.typing.MatLike): 마스크 이미지

    Returns:
        Union[Tuple[int, int], None]: 무게중심 좌표 또는 None (유효한 마스크가 없는 경우)
    """
    # 마스크의 모멘트를 계산합니다.
    moments = cv2.moments(mask, binaryImage=True)

    if moments["m00"] == 0:
        return None

    # 무게중심 좌표를 계산합니다.
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])

    return center_x, center_y


def get_common_area(mask: cv2.typing.MatLike, line: Tuple[Tuple[int, int], Tuple[int, int]]) -> cv2.typing.MatLike:
    """
    선과 마스크가 겹치는 공통 부분을 찾습니다.

    Args:
        mask (cv2.typing.MatLike): 마스크 이미지
        line (Tuple[Tuple[int, int], Tuple[int, int]]): 선 좌표

    Returns:
        cv2.typing.MatLike: 공통 부분
    """
    # 마스크와 동일한 크기의 빈 마스크를 생성합니다.
    common_mask = numpy.zeros_like(mask, dtype=numpy.uint8)
    # 빈 마스크에 선을 그립니다.
    cv2.line(common_mask, line[0], line[1], 255, 5)
    # 원래 마스크와 선이 그려진 마스크의 공통 부분을 계산합니다.
    common_mask = cv2.bitwise_and(mask, common_mask)

    return common_mask
