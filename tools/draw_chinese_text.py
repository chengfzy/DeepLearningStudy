"""
Draw Chinese Text on Image
"""

import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    # argument
    parser = argparse.ArgumentParser(description='Draw Chinese text on image')
    parser.add_argument('--image', required=True, help='image file')
    parser.add_argument('--text', default='你好中国 Hello China!', help='text')
    args = parser.parse_args()
    print(args)

    # some parameters
    rect_points = np.array([[[184, 66], [320, 64], [340, 170], [160, 180]]], dtype=np.int32)

    # read image and resize if it's too large
    img_raw = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if max(img_raw.shape) > 500:
        ratio = 500 / max(img_raw.shape)
        img_raw = cv2.resize(img_raw, (0, 0), fx=ratio, fy=ratio)

    # draw rect
    cv2.polylines(img_raw, rect_points, isClosed=True, color=(0, 255, 0))

    # convert image from OpenCV to PIL
    img_pil = Image.fromarray(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
    # draw chinese text
    font = ImageFont.truetype('NotoSansCJK-Regular.ttc', 15)
    draw = ImageDraw.Draw(img_pil)
    draw.text(rect_points[0, 0], args.text, font=font, fill=(0, 255, 255))
    # convert to OpenCV
    img_result = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

    # show result
    cv2.imshow('Result', img_result)
    cv2.waitKey()


if __name__ == '__main__':
    main()
