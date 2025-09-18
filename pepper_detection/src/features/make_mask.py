import os
import cv2
import numpy as np

x_pos = -1
y_pos = -1
is_dragging = False

size = 10


def mouse_callback(event, x, y, flags, param):
    global x_pos, y_pos, is_dragging

    if event == cv2.EVENT_LBUTTONDOWN:
        is_dragging = True
        x_pos = x
        y_pos = y
        print(f"Mouse clicked at ({x}, {y})")

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_dragging:
            x_pos = x
            y_pos = y

    elif event == cv2.EVENT_LBUTTONUP:
        if is_dragging:
            is_dragging = False
            print(f"Mouse released at ({x}, {y})")


def make_mask(image_file: str):
    global x_pos, y_pos, size

    image_file_path = os.path.join(os.getcwd(), f"gan/input/{image_file}")

    img = cv2.imread(image_file_path)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_callback)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    while True:
        cv2.imshow("image", img)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break

        if x_pos != -1 and y_pos != -1:
            cv2.circle(mask, (x_pos, y_pos), size, (255, 255, 255), -1)
            cv2.circle(img, (x_pos, y_pos), size, (0, 0, 255), -1)

            x_pos = -1
            y_pos = -1

    cv2.destroyAllWindows()

    mask_file_path = os.path.join(os.getcwd(), f"gan/mask/{image_file}")
    cv2.imwrite(mask_file_path, mask)
