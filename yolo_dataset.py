# Author: Ming Yang
# Date: 2023/01/20
# Description: Traverse the zip file but not decompress it.
# Suppose the zip file contains yolo format annotation files.
# /
# ├── classes.txt
# ├── images
# │   ├── 1.jpg
# │   ├── 2.jpg
# │   └── ...
# └── labels
#     ├── 1.txt
#     ├── 2.txt
#     └── ...
from datetime import datetime
from typing import Optional
from zipfile import ZipFile

from PIL import Image
from pycocotools import coco
from pycocotools.coco import COCO

yolo_label = {
    'class': str,
    'class_id': int,
    'x_center': float,
    'y_center': float,
    'width': float,
    'height': float
}

yolo_image = {
    'image_name': str,
    'image': Image,
    'labels': list[yolo_label]
}

coco_annotation = {
    "id": int,
    "image_id": int,  # the id of the image that the annotation belongs to
    "category_id": int,  # the id of the category that the annotation belongs to
    # "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [float, float, float, float],  # [x,y,width,height]
    "iscrowd": bool,  # 0 or 1,
}

coco_category = {
    "id": int,
    "name": str,
    "supercategory": Optional[str],
}

coco_image = {
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
    "date_captured": Optional[datetime],
}

coco_dataset = {
    "images": list[coco_image],  # list of all images in the dataset
    "annotations": list[coco_annotation],  # list of all annotations in the dataset
    "categories": list[coco_category]  # list of all categories
}


class YoloImage:
    def __init__(self, image_name: str, image: Image, labels: list[yolo_label]):
        self.image_name = image_name
        self.image = image
        self.labels = labels

    def __repr__(self):
        return f'YoloImage(image_name={self.image_name}, image={self.image}, labels={self.labels})'

    def to_coco_image(self, id: int) -> coco_image:
        return {
            "id": id,
            "width": self.image.width,
            "height": self.image.height,
            "file_name": self.image_name,
        }

    def to_coco_annotations(self, image_id: int, ann_id_start: int) -> list[coco_annotation]:
        ann_id = ann_id_start
        annotations: list[coco_annotation] = []
        for label in self.labels:
            ann_id = ann_id + 1
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": label['class_id'],
                "area": label['width'] * label['height'],
                "bbox": [label['x_center'] - label['width'] / 2, label['y_center'] - label['height'] / 2,
                         label['width'], label['height']],
                "iscrowd": False,
            })
        return annotations


class YoloDataset:
    _zip_file: ZipFile
    _classes: list[str]
    _images: list[str]
    _labels: list[str]

    def __init__(self, zip_file: ZipFile, classes=None, images=None, labels=None):
        if labels is None:
            labels = []
        if images is None:
            images = []
        if classes is None:
            classes = []
        self._zip_file = zip_file
        self._classes = classes
        self._images = images
        self._labels = labels

    @staticmethod
    def from_zip_file(zip_file: ZipFile) -> 'YoloDataset':
        namelist = zip_file.namelist()
        root_name = namelist[0]
        namelist = list(filter(lambda x: not zip_file.getinfo(x).is_dir(), namelist))
        if 'classes.txt' in namelist:
            classes = zip_file.read('classes.txt').decode('utf-8').split('\n')
        else:
            classes = []
        images = list(filter(lambda x: x.startswith(root_name + 'images'), namelist))
        labels = list(filter(lambda x: x.startswith(root_name + 'labels'), namelist))
        assert len(images) == len(labels) and len(images) > 0
        images.sort()
        labels.sort()
        for image, label in zip(images, labels):
            image_name = image.split('/')[-1]
            label_name = label.split('/')[-1]
            assert image_name.split('.')[0] == label_name.split('.')[0]
        return YoloDataset(zip_file, classes, images, labels)

    @staticmethod
    def from_path(path: str) -> 'YoloDataset':
        zip_file = ZipFile(path, 'r')
        return YoloDataset.from_zip_file(zip_file)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index: int) -> YoloImage:
        image_name = self._images[index]
        labels = self._zip_file.read(self._labels[index]).decode('utf-8').split('\n')
        labels = list(filter(lambda x: len(x) > 0, labels))
        labels = list(map(lambda x: x.split(' '), labels))
        labels = list(map(lambda x: {
            'class': self._classes[int(x[0])] if len(self._classes) > int(x[0]) else 'unknown',
            'class_id': int(x[0]),
            'x_center': float(x[1]),
            'y_center': float(x[2]),
            'width': float(x[3]),
            'height': float(x[4])
        }, labels))

        return YoloImage(image_name, Image.open(self._zip_file.open(self._images[index])), labels)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __deepcopy__(self, memodict=None):
        return YoloDataset(self._zip_file, self._classes, self._images, self._labels)

    def load_image(self, image_name: str) -> Image:
        return Image.open(self._zip_file.open(image_name))

    def to_coco(self) -> COCO:
        images: list[coco_image] = []
        annotations: list[coco_annotation] = []
        categories: list[coco_category] = []
        ann_id = 0
        for i in range(len(self)):
            image = self[i]
            images.append(image.to_coco_image(i))
            annotations.extend(image.to_coco_annotations(i, ann_id))
            ann_id = ann_id + len(image.labels)
        for i in range(len(self._classes)):
            categories.append({
                "id": i,
                "name": self._classes[i],
                "supercategory": None,
            })

        coco_ds = coco.COCO()
        coco_ds.dataset = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        coco_ds.createIndex()
        return coco_ds


if __name__ == '__main__':
    dataset = YoloDataset.from_zip_file('tests/coco8.zip')
    coco = dataset.to_coco()
    print(coco)
