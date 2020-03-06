import imageio
import numpy as np
from skimage.transform import resize


def read_img_from_path(path):
    img = imageio.imread(path, pilmode="RGB")
    return img


def read_from_file(file_object):
    img = imageio.imread(file_object, pilmode="RGB")

    return img


def resize_img(img, h=224, w=224):
    desired_size_h = h
    desired_size_w = w

    old_size = img.shape[:2]

    ratio = min(desired_size_w / old_size[1], desired_size_h / old_size[0])

    new_size = tuple([int(x * ratio) for x in old_size])

    im = resize(img, (new_size[0], new_size[1]))

    delta_w = desired_size_w - new_size[1]
    delta_h = desired_size_h - new_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    new_im = np.ones((h, w, im.shape[-1])) * 255
    new_im[top : (im.shape[0] + top), left : (im.shape[1] + left), :] = im

    return new_im
