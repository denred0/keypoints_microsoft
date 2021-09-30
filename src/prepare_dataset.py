import shutil
import json
import xml.etree.cElementTree as ET

import cv2
from tqdm import tqdm
from pathlib import Path

from my_utils import get_all_files_in_folder


def prepare_train(resize_size=(512, 512)):
    dirpath_images = Path('data/train/images')
    if dirpath_images.exists() and dirpath_images.is_dir():
        shutil.rmtree(dirpath_images)
    Path(dirpath_images).mkdir(parents=True, exist_ok=True)

    dirpath_annot = Path('data/train/annotations')
    if dirpath_annot.exists() and dirpath_annot.is_dir():
        shutil.rmtree(dirpath_annot)
    Path(dirpath_annot).mkdir(parents=True, exist_ok=True)

    images = sorted(get_all_files_in_folder(Path('data/source/train_val_images'), ['*.jpg']))
    annotations = sorted(get_all_files_in_folder(Path('data/source/train_val_annot'), ['*.json']))

    if len(images) != len(annotations):
        raise ValueError("Count of images is not equal count of labels")

    for im_path, annot_path in tqdm(zip(images, annotations), total=len(images), desc='Preparing train'):
        assert im_path.stem == annot_path.stem, 'Image and annot have different names'

        img = cv2.imread(str(im_path), cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image_res = cv2.resize(img, resize_size, interpolation=cv2.INTER_NEAREST)

        x_scale_factor = resize_size[0] / img.shape[1]
        y_scale_factor = resize_size[1] / img.shape[0]

        with open(annot_path) as j:
            json_ = json.load(j)

        target_points_dict = json_["object"]
        exist_bad_keypoints = False

        for v in target_points_dict.values():
            if v[0] < 0 or v[1] < 0:
                exist_bad_keypoints = True
                break

            v[0] = v[0] * x_scale_factor
            v[1] = v[1] * y_scale_factor

        if exist_bad_keypoints:
            continue

        # save image
        cv2.imwrite(str(Path.joinpath(dirpath_images, im_path.name)), image_res)

        # create xml annotation file
        root = ET.Element("annotation")

        ET.SubElement(root, "folder").text = "images"
        ET.SubElement(root, "filename").text = im_path.name
        ET.SubElement(root, "path").text = '../images/' + im_path.name

        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = 'Unknown'

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(resize_size[0])
        ET.SubElement(size, "height").text = str(resize_size[1])
        ET.SubElement(size, "depth").text = str(img.shape[2])

        ET.SubElement(root, "segmented").text = '0'

        object = ET.SubElement(root, "object")
        ET.SubElement(object, "name").text = 'person'
        ET.SubElement(object, "pose").text = 'Unspecified'
        ET.SubElement(object, "truncated").text = '0'
        ET.SubElement(object, "difficult").text = '0'

        bndbox = ET.SubElement(object, "bndbox")
        ET.SubElement(bndbox, "xmin").text = '0'
        ET.SubElement(bndbox, "ymin").text = '0'
        ET.SubElement(bndbox, "xmax").text = str(resize_size[0])
        ET.SubElement(bndbox, "ymax").text = str(resize_size[1])

        keypoints = ET.SubElement(object, "keypoints")
        for key, value in target_points_dict.items():
            point = ET.SubElement(keypoints, key)
            ET.SubElement(point, "x").text = str(int(value[0]))
            ET.SubElement(point, "y").text = str(int(value[1]))

        tree = ET.ElementTree(root)
        tree.write(Path.joinpath(dirpath_annot, annot_path.stem + '.xml'))


def prepare_test(resize_size=(512, 512)):
    dirpath_images = Path('data/test/images')
    if dirpath_images.exists() and dirpath_images.is_dir():
        shutil.rmtree(dirpath_images)
    Path(dirpath_images).mkdir(parents=True, exist_ok=True)

    images = sorted(get_all_files_in_folder(Path('data/source/test'), ['*.jpg']))

    for im_path in tqdm(images, desc='Preparing test'):
        img = cv2.imread(str(im_path), cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image_res = cv2.resize(img, resize_size, interpolation=cv2.INTER_NEAREST)

        # save image
        cv2.imwrite(str(Path.joinpath(dirpath_images, im_path.name)), image_res)


if __name__ == '__main__':
    resize_size = (512, 512)
    prepare_train(resize_size=resize_size)
    prepare_test(resize_size=resize_size)
