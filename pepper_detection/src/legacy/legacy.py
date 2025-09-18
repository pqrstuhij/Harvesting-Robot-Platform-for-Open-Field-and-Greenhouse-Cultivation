import cv2
import numpy


def apply_xyn_mask_to_image(image: cv2.UMat, xyn: numpy.ndarray) -> cv2.UMat:
    """
    정규화된 마스크를 이미지에 적용합니다.

    Args:
        image (cv2.UMat): 이미지
        xyn (numpy.ndarray): 정규화된 마스크 데이터 (0 ~ 1)

    Returns:
        cv2.UMat: 마스크가 적용된 이미지
    """
    x_pixels = xyn[:, 0] * image.shape[1]
    y_pixels = xyn[:, 1] * image.shape[0]

    mask = numpy.zeros((image.shape[0], image.shape[1]), numpy.uint8)

    points = numpy.array(list(zip(x_pixels, y_pixels)), numpy.int32)
    cv2.fillPoly(mask, [points], 255)

    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image


def get_red_area(image: cv2.UMat, lower: int = 100, upper: int = 255) -> cv2.UMat:
    """
    빨간색 영역을 반환합니다.

    Args:
        image (cv2.UMat): 이미지
        lower (int, optional): 빨간색 하한값. Defaults to 100.
        upper (int, optional): 빨간색 상한값. Defaults to 255.

    Returns:
        cv2.UMat: 빨간색 영역
    """
    lower_red = numpy.array([0, 0, lower])
    upper_red = numpy.array([100, 100, upper])

    mask = cv2.inRange(image, lower_red, upper_red)
    red_area = cv2.bitwise_and(image, image, mask=mask)
    return red_area


def erode_operation(image: cv2.UMat, kernel: cv2.typing.MatLike = (3, 3)) -> cv2.UMat:
    """
    침식 연산을 수행합니다.

    Args:
        image (cv2.UMat): 이미지
        kernel (cv2.typing.MatLike, optional): 커널. Defaults to (3, 3).

    Returns:
        cv2.UMat: 침식 연산 결과 이미지
    """
    erosion = cv2.erode(image, kernel, iterations=1)
    return erosion


def dilate_operation(image: cv2.UMat, kernel: cv2.typing.MatLike = (3, 3)) -> cv2.UMat:
    """
    팽창 연산을 수행합니다.

    Args:
        image (cv2.UMat): 이미지
        kernel (cv2.typing.MatLike, optional): 커널. Defaults to (3, 3).

    Returns:
        cv2.UMat: 팽창 연산 결과 이미지
    """
    dilation = cv2.dilate(image, kernel, iterations=1)
    return dilation


def close_operation(image: cv2.UMat, kernel: cv2.typing.MatLike = (3, 3)) -> cv2.UMat:
    """
    닫힘 연산을 수행합니다.

    Args:
        image (cv2.UMat): 이미지

    Returns:
        cv2.UMat: 닫힘 연산 결과 이미지
    """
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closing


def get_largest_contour(image: cv2.UMat) -> numpy.ndarray:
    """
    가장 큰 컨투어를 반환합니다.

    Args:
        image (cv2.UMat): 이미지

    Returns:
        numpy.ndarray: 가장 큰 컨투어
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    cv2.imshow("thresh", thresh)
    cv2.waitKey(0)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour


def draw_area(image, label):
    for i in range(1, len(label)):
        x1 = int(label[i - 1][0] * image.shape[1])
        y1 = int(label[i - 1][1] * image.shape[0])

        x2 = int(label[i][0] * image.shape[1])
        y2 = int(label[i][1] * image.shape[0])

        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return image


def apply_transparent_mask(image, binary_mask, color=(0, 200, 200), alpha=0.5):
    """
    이미지의 지정된 마스크 영역에 지정한 색으로 반투명 처리를 적용합니다.

    Args:
        image (np.ndarray): 원본 이미지
        binary_mask (np.ndarray): 이진 마스크 (0과 255 값)

    Returns:
        np.ndarray: 노란색 반투명 효과가 적용된 이미지
    """
    overlay = image.copy()  # 오버레이 이미지를 원본 복사
    y_coords, x_coords = numpy.where(binary_mask > 0)

    # 노란색 영역 추가
    for y, x in zip(y_coords, x_coords):
        overlay[y, x] = [color[0], color[1], color[2]]

    # 반투명 효과 적용
    alpha = 0.5  # 투명도 설정
    blended_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return blended_image


def draw_bounding_boxes_with_labels(image, contours, labelText, labelColor=(0, 0, 255)):
    """
    이미지에 바운딩 박스와 텍스트 레이블을 추가합니다.

    Args:
        image (np.ndarray): 원본 이미지
        contours (list): 윤곽선 리스트
        labelText (str): 텍스트 레이블
        labelColor (tuple): 사각형 색상 (BGR)

    Returns:
        np.ndarray: 바운딩 박스와 텍스트가 추가된 이미지
    """
    for contour in contours:
        # 윤곽선에서 바운딩 박스 좌표 계산
        x, y, w, h = cv2.boundingRect(contour)

        # 바운딩 박스 그리기
        cv2.rectangle(image, (x, y), (x + w, y + h), labelColor, 2)

        # 텍스트 크기 계산
        (text_width, text_height), baseline = cv2.getTextSize(
            labelText, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # 텍스트를 감싸는 사각형 계산
        text_bg_x1, text_bg_y1 = x, y - text_height - baseline
        text_bg_x2, text_bg_y2 = x + text_width, y

        # 텍스트 사각형이 이미지 위쪽을 벗어날 경우 조정
        if text_bg_y1 < 0:
            text_bg_y1 = y + h
            text_bg_y2 = text_bg_y1 + text_height + baseline

        # 텍스트 배경 사각형 그리기
        cv2.rectangle(image, (text_bg_x1, text_bg_y1),
                      (text_bg_x2, text_bg_y2), labelColor, -1)

        # 흰색 텍스트 추가
        cv2.putText(
            image, labelText, (text_bg_x1, text_bg_y2 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    return image
