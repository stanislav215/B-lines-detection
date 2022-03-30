# @title augmentation functions
from skimage.transform import rotate, EuclideanTransform, warp
import cv2

from diploma import *

def horizontal_flip(p=0.5):
    def fun(image, p=p):
        if np.random.rand() < p:
            image = image[:, ::-1, :]
        return image
    return fun
def vertical_flip(p=0.5):
    def fun(image, p=p):
        if np.random.rand() < p:
            image = image[::-1, :, :]
        return image
    return fun
def random_rotation(angle_range=(0, 180), p=0.5):
    def fun(image, angle_range=angle_range, p=p):
        if np.random.rand() < p:
            h, w, _ = image.shape
            angle = np.random.randint(*angle_range)
            image = rotate(image, angle)
        return image
    return fun
def random_erease(mask_size_range=(0,30), mask_value='mean', constant_value=0, p=0.5):
    def fun(image_origin, mask_size_range=mask_size_range, mask_value=mask_value, constant_value=constant_value, p=p):
        if np.random.rand() < p:
            image = np.copy(image_origin)
            mask_size = np.random.randint(*mask_size_range)
            if mask_value == 'mean':
                mask_value = image.mean()
            elif mask_value == 'random':
                mask_value = np.random.randint(0, 256)
            elif mask_value == 'constant':
                mask_value = constant_value

            h, w, _ = image.shape
            top = np.random.randint(0 - mask_size // 2, h - mask_size)
            left = np.random.randint(0 - mask_size // 2, w - mask_size)
            bottom = top + mask_size
            right = left + mask_size
            if top < 0:
                top = 0
            if left < 0:
                left = 0
            image[top:bottom, left:right, :].fill(mask_value)
            return image
        return image_origin
    return fun

def random_crop(crop_size_width=0.9,crop_size_height=0.9,p=0.5):
    def fun(image, crop_size_width=crop_size_width, crop_size_height=crop_size_height, p=p):
        if np.random.rand() < p:
            crop_size = (int(image.shape[0]*crop_size_height),int(image.shape[1]*crop_size_width))
            h, w, c = image.shape
            top = np.random.randint(0, h - crop_size[0])
            left = np.random.randint(0, w - crop_size[1])
            bottom = top + crop_size[0]
            right = left + crop_size[1]
            image = image[top:bottom, left:right, :]
            image = cv2.resize(image, (h,w),  interpolation=cv2.INTER_LINEAR)
            image = image.reshape(h,w,c)
        return image
    return fun

def random_translation_and_rotation(width_range=(-0.1,0.1),heigth_range=(-0.1,0.1), angle_range=(0,10), p=0.5):
    def fun(image, width_range=width_range,heigth_range=heigth_range, angle_range=angle_range, p=p):
        if np.random.rand() < p:
            random_width = np.random.uniform(*width_range)*image.shape[1]
            random_height = np.random.uniform(*heigth_range)*image.shape[0]
            angle = np.radians(np.random.randint(*angle_range))
            rotation = EuclideanTransform(rotation=angle)
            shift = EuclideanTransform(translation=-np.array(image.shape[:2]) / 2)
            matrix = np.linalg.inv(shift.params) @ rotation.params @ shift.params
            tform = EuclideanTransform(matrix)
            return warp(image, tform.inverse)
        return image
    return fun
def random_contrast(factor=0.3, p=0.5):
    def fun(image, factor=factor, p=p):
        if np.random.rand() < p:
            lower = upper = factor
            image = tf.image.random_contrast(image, 1. - lower, 1. + upper).numpy()
        return image
    return fun