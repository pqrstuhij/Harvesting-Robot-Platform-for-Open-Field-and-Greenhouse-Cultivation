import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


image_path = os.path.join(os.getcwd(), "data/pepper/train/images")[:100]
label_path = os.path.join(os.getcwd(), "data/pepper/train/labels")[:100]

image_files = os.listdir(image_path)


def mask_label(image, label):
    mask = np.zeros(image.shape, dtype=np.uint8)

    x_pixels = [int(x * image.shape[1]) for x, y in label]
    y_pixels = [int(y * image.shape[0]) for x, y in label]

    points = np.array(list(zip(x_pixels, y_pixels)), np.int32)
    cv2.fillPoly(mask, [points], [255, 255, 255])

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


h_freq = np.zeros(180, dtype=np.int64)

for image_file in image_files:
    image = cv2.imread(os.path.join(image_path, image_file))
    label_file = image_file.replace(".jpg", ".txt")

    layer_list = []
    with open(os.path.join(label_path, label_file), "r") as f:
        lines = f.readlines()
        for line in lines:
            points = line.strip().split(" ")

            label = []
            for i in range(1, len(points), 2):
                x = float(points[i])
                y = float(points[i + 1])

                label.append((x, y))

            layer_list.append(label)

    mask = np.zeros(image.shape, dtype=np.uint8)
    for label in layer_list:
        temp = mask_label(image.copy(), label)
        mask = cv2.bitwise_or(mask, temp)

    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)
    h = h.flatten()
    freq, bins = np.histogram(h, bins=180, range=[0, 180])

    h_freq += freq


# h_freq = h_freq / h_freq.sum()

for i, freq in enumerate(h_freq):
    print(f"{i}: {freq}")

f, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True)
d = .7

kwargs = dict(marker=[(-1, -d), (1, d)], markersize=15,
              linestyle='none', color='k', clip_on=False)

ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax1.xaxis.set_ticks_position('none')

ax1.yaxis.grid()
ax2.yaxis.grid()

ax1.set_ylim(850000000, 870000000)
ax2.set_ylim(0, 5000000)


# 데이터를 그래프에 표시
ax1 = plt.subplot(211)
ax1.plot(h_freq)
ax2 = plt.subplot(212)
ax2.plot(h_freq)

plt.show()

# lower_hue1 = np.array([0, 0, 0])
# upper_hue1 = np.array([9, 255, 255])

# lower_hue2 = np.array([167, 0, 0])
# upper_hue2 = np.array([179, 255, 255])

# mask1 = cv2.inRange(hsv, lower_hue1, upper_hue1)
# mask2 = cv2.inRange(hsv, lower_hue2, upper_hue2)
# h_mask = cv2.bitwise_or(mask1, mask2)

# h_mask_image = cv2.bitwise_and(mask, mask, mask=h_mask)

# cv2.imshow("mask", mask)
# cv2.imshow("h_mask_image", h_mask_image)
# cv2.waitKey(0)
