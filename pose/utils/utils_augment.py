#  Random occlusion code are adapted from https://github.com/isarandi/synthetic-occlusion/blob/master/augmentation.py

import os.path
import random
import xml.etree.ElementTree
import numpy as np
import matplotlib.pyplot as plt
import skimage.data
import cv2
import PIL.Image


class Augment(object):
    def __init__(self, aug_rate = 0.5, num_occluder=8, datatype = 'animal'):
        self.obj_path = './animal_data/VOCdevkit/VOC2012'
        self.datatype = datatype
        self.occluders = self.load_occluders()
        self.aug_rate = aug_rate
        self.num_occluder = num_occluder

    def augment_occlu(self, img):
        aug = 0.0
        if np.random.uniform(0, 1) < self.aug_rate:
            img = self.occlude_with_objects(img)
            aug = 1.
        return img, aug

    def load_occluders(self):
        occluders = []
        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))

        annotation_paths = list_filepaths(os.path.join(self.obj_path, 'Annotations'))
        # category = []
        for annotation_path in annotation_paths:
            xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
            is_segmented = (xml_root.find('segmented').text != '0')

            if not is_segmented:
                continue

            boxes = []

            for i_obj, obj in enumerate(xml_root.findall('object')):
                # category.append(obj.find('name').text)
                is_cat = (obj.find('name').text == 'cat')
                is_dog = (obj.find('name').text == 'dog')
                is_cow = (obj.find('name').text == 'cow')
                is_horse = (obj.find('name').text == 'horse')
                is_sheep = (obj.find('name').text == 'sheep')
                is_human = (obj.find('name').text == 'person')
                is_animal = is_cat or is_dog or is_cow or is_horse or is_sheep or is_human
                # to_delete = is_animal if self.datatype == 'animal' else is_human
                to_delete = is_animal
                if not to_delete:
                    bndbox = obj.find('bndbox')
                    box = [int(bndbox.find(s).text) for s in ['xmin', 'ymin', 'xmax', 'ymax']]
                    boxes.append((i_obj, box))

            if not boxes:
                continue

            im_filename = xml_root.find('filename').text
            seg_filename = im_filename.replace('jpg', 'png')

            im_path = os.path.join(self.obj_path, 'JPEGImages', im_filename)
            seg_path = os.path.join(self.obj_path, 'SegmentationObject', seg_filename)

            im = np.asarray(PIL.Image.open(im_path))
            labels = np.asarray(PIL.Image.open(seg_path))

            for i_obj, (xmin, ymin, xmax, ymax) in boxes:
                object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(np.uint8) * 255
                object_image = im[ymin:ymax, xmin:xmax]
                if cv2.countNonZero(object_mask) < 500:
                    # Ignore small objects
                    continue

                # Reduce the opacity of the mask along the border for smoother blending
                eroded = cv2.erode(object_mask, structuring_element)
                object_mask[eroded < object_mask] = 192
                object_with_mask = np.concatenate([object_image, object_mask[..., np.newaxis]], axis=-1)

                # Downscale for efficiency
                object_with_mask = resize_by_factor(object_with_mask, 0.5)
                object_with_mask = object_with_mask.astype(np.float32) / 255.0
                occluders.append(object_with_mask)
        # print(set(category))
        return occluders

    def load_images(self):
        occluders = []
        annotation_paths = list_filepaths(os.path.join(self.obj_path, 'Annotations'))
        for annotation_path in annotation_paths:
            valid = True
            xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
            is_segmented = (xml_root.find('segmented').text != '0')

            if not is_segmented:
                continue
            for i_obj, obj in enumerate(xml_root.findall('object')):
                # print(obj.find('name').text)
                is_cat = (obj.find('name').text == 'cat')
                is_dog = (obj.find('name').text == 'dog')
                is_cow = (obj.find('name').text == 'cow')
                is_horse = (obj.find('name').text == 'horse')
                is_sheep = (obj.find('name').text == 'sheep')
                if is_cat or is_dog or is_cow or is_horse or is_sheep:
                    valid = False
                    break
            if valid:
                im_filename = xml_root.find('filename').text
                im_path = os.path.join(self.obj_path, 'JPEGImages', im_filename)
                occluders.append(im_path)
        return occluders

    def occlude_with_objects(self, im):
        """Returns an augmented version of `im`, containing some occluders from the Pascal VOC dataset."""

        result = im.copy()
        width_height = np.asarray([im.shape[1], im.shape[0]])
        count = np.random.randint(1, self.num_occluder)
        for _ in range(count):
            occluder = random.choice(self.occluders)
            random_scale_factor = np.random.uniform(0.2, 0.8)
            scale_factor = random_scale_factor
            occluder = resize_by_factor(occluder, scale_factor)

            center = np.random.uniform([0, 0], width_height)
            self.paste_over(im_src=occluder, im_dst=result, center=center)

        return result

    def paste_over(self, im_src, im_dst, center):
        """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.
        Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
        `im_src` becomes visible).
        Args:
            im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
            im_dst: The target image.
            alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
                at each pixel. Large values mean more visibility for `im_src`.
            center: coordinates in `im_dst` where the center of `im_src` should be placed.
        """

        width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
        width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])

        center = np.round(center).astype(np.int32)
        raw_start_dst = center - width_height_src // 2
        raw_end_dst = raw_start_dst + width_height_src

        start_dst = np.clip(raw_start_dst, 0, width_height_dst)
        end_dst = np.clip(raw_end_dst, 0, width_height_dst)
        region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

        start_src = start_dst - raw_start_dst
        end_src = width_height_src + (end_dst - raw_end_dst)
        region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
        color_src = region_src[..., 0:3]
        alpha = region_src[..., 3:].astype(np.float32)

        im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (
                alpha * color_src + (1 - alpha) * region_dst)


def resize_by_factor(im, factor):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    """
    new_size = tuple(np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
    interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)


def list_filepaths(dirpath):
    names = os.listdir(dirpath)
    paths = [os.path.join(dirpath, name) for name in names]
    return sorted(filter(os.path.isfile, paths))
