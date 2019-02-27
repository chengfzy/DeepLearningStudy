"""
Draw Chinese Text on Image
"""

import argparse
import enum

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Color(enum.Enum):
    """Color Definitions"""
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    YELLOW = (0, 255, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)


def pil_color(color: Color):
    """Get PIL color from Color definitions"""
    return color.value[2], color.value[1], color.value[0]


def draw_poly():
    """draw poly and Chinese text on image"""
    # argument
    parser = argparse.ArgumentParser(description='Draw Chinese text on image')
    parser.add_argument('--image', required=True, help='image file')
    parser.add_argument('--text', default='你好中国 Hello China!', help='text')
    args = parser.parse_args()
    print(args)

    # some parameters
    rect_points = np.array([[[184, 66], [320, 64], [340, 170], [160, 180]]], dtype=np.int32)

    # read image and resize if it's too large
    img_raw = cv.imread(args.image, cv.IMREAD_UNCHANGED)
    if max(img_raw.shape) > 500:
        ratio = 500 / max(img_raw.shape)
        img_raw = cv.resize(img_raw, (0, 0), fx=ratio, fy=ratio)

    # draw poly
    cv.polylines(img_raw, rect_points, isClosed=True, color=Color.GREEN.value)

    # convert image from OpenCV to PIL
    img_pil = Image.fromarray(cv.cvtColor(img_raw, cv.COLOR_BGR2RGB))
    # draw chinese text
    font = ImageFont.truetype('NotoSansCJK-Regular.ttc', 15)
    draw = ImageDraw.Draw(img_pil)
    draw.text(rect_points[0, 0], args.text, font=font, fill=pil_color(Color.CYAN))
    # convert to OpenCV
    img_result = cv.cvtColor(np.asarray(img_pil), cv.COLOR_RGB2BGR)

    # show result
    cv.imshow('Result', img_result)
    cv.waitKey()


def draw_detections():
    # argument
    parser = argparse.ArgumentParser(description='Draw Chinese text on image')
    parser.add_argument('--image', required=True, help='image file')
    parser.add_argument('--text', default='你好中国 Hello China!', help='text')
    args = parser.parse_args()
    print(args)

    # some parameters
    pt1 = (90, 30)
    pt2 = (340, 170)

    # read image and resize if it's too large
    img_raw = cv.imread(args.image, cv.IMREAD_UNCHANGED)
    if max(img_raw.shape) > 500:
        ratio = 500 / max(img_raw.shape)
        img_raw = cv.resize(img_raw, (0, 0), fx=ratio, fy=ratio)

    # draw rect
    cv.rectangle(img_raw, pt1, pt2, Color.GREEN.value, thickness=1, lineType=cv.LINE_AA)

    # convert image from OpenCV to PIL
    img_pil = Image.fromarray(cv.cvtColor(img_raw, cv.COLOR_BGR2RGB))
    # draw chinese text
    font = ImageFont.truetype('NotoSansCJK-Regular.ttc', 15)
    draw = ImageDraw.Draw(img_pil, "RGBA")
    # text size
    text_size = draw.textsize(args.text, font=font)
    # draw text and rect
    draw.rectangle((pt1, (pt1[0] + text_size[0], pt1[1] + text_size[1])), fill=pil_color(Color.GREEN) + (150,))
    draw.text(pt1, args.text, font=font, fill=pil_color(Color.WHITE))
    # convert to OpenCV
    img_result = cv.cvtColor(np.asarray(img_pil), cv.COLOR_RGB2BGR)

    # show result
    cv.imshow('Result', img_result)
    cv.waitKey()


if __name__ == '__main__':
    # draw_poly()
    draw_detections()
