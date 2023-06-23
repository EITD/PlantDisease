import cv2
import os
import numpy
import random
from skimage import exposure
from paddle.vision.transforms import RandomRotation
from paddle.vision.transforms import CenterCrop
from paddle.vision.transforms import Resize


# def cropImage(arrimg, dst):
#     crop = CenterCrop(224)
#     cropimg = crop(arrimg)
#     cv2.imwrite(dst, cropimg)


# def convert2brightness(arrimg, dst):
#     transform = ColorJitter(0.4)
#     brightnessimg = transform(arrimg)
#     cv2.imwrite(dst, brightnessimg * 255)
#
#
# def convert2contrast(arrimg, dst):
#     transform = ColorJitter(0, 0.4)
#     contrastimg = transform(arrimg)
#     cv2.imwrite(dst, contrastimg * 255)


def convert2gamma(arrimg, dst):
    gamma = random.uniform(0.5, 1.5)
    if numpy.any(arrimg < 0):
        arrimg = exposure.rescale_intensity(arrimg, in_range=(0, 1), out_range=(0, 1))
    gammaimg = exposure.adjust_gamma(arrimg, gamma)
    cv2.imwrite(dst, gammaimg * 255)


def convert2log(arrimg, dst):
    if numpy.any(arrimg < 0):
        arrimg = exposure.rescale_intensity(arrimg, in_range=(0, 1), out_range=(0, 1))
    logimg = exposure.adjust_log(arrimg)
    cv2.imwrite(dst, logimg * 255)


def convert2adapthist(arrimg, dst):
    adapthistimg = exposure.equalize_adapthist(arrimg)
    cv2.imwrite(dst, adapthistimg * 255)


def flipImage(arrimg, dst):
    flipimg = cv2.flip(arrimg, -1)
    cv2.imwrite(dst, flipimg * 255)


def horizontalImage(arrimg, dst):
    horizontalimg = cv2.flip(arrimg, 1)
    cv2.imwrite(dst, horizontalimg * 255)


def rotateImage(arrimg, dst):
    rotate = RandomRotation(60)
    rotateimg = rotate(arrimg)
    crop = CenterCrop(163)
    cropimg = crop(rotateimg)
    resize = Resize(224)
    cv2.imwrite(dst, resize(cropimg) * 255)


def normalizeImage(arrimg):
    normalizeImg = numpy.zeros_like(numpy.float32(arrimg))
    cv2.normalize(arrimg, normalizeImg, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return normalizeImg


if __name__ == '__main__':
    for root1, dirs1, files1 in os.walk("../dataset/data_image"):
        for dir in dirs1:
            for root2, dirs2, files2 in os.walk("../dataset/data_image/" + dir):
                for file in files2:
                    src = os.path.join(root2, file)
                    arrimg = cv2.imread(src)
                    crop = CenterCrop(224)
                    cropimg = crop(arrimg)
                    dst = os.path.join(root2, file)
                    cv2.imwrite(dst, cropimg)
                    normalizeImg = normalizeImage(cropimg)
                    dst1 = os.path.join(root2, "gamma_" + file)
                    convert2gamma(normalizeImg, dst1)
                    dst2 = os.path.join(root2, "log_" + file)
                    convert2log(normalizeImg, dst2)
                    dst3 = os.path.join(root2, "adapthist_" + file)
                    convert2adapthist(normalizeImg, dst3)
                    dst4 = os.path.join(root2, "flip_" + file)
                    flipImage(normalizeImg, dst4)
                    dst5 = os.path.join(root2, "horizontal_" + file)
                    horizontalImage(normalizeImg, dst5)
                    dst6 = os.path.join(root2, "rotate_" + file)
                    rotateImage(normalizeImg, dst6)
