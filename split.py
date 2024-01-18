from urllib.parse import urlparse
import boto3
from io import BytesIO


class S3Url(object):
    """
    >>> s = S3Url("s3://bucket/hello/world")
    >>> s.bucket
    'bucket'
    >>> s.key
    'hello/world'
    >>> s.url
    's3://bucket/hello/world'

    >>> s = S3Url("s3://bucket/hello/world?qwe1=3#ddd")
    >>> s.bucket
    'bucket'
    >>> s.key
    'hello/world?qwe1=3#ddd'
    >>> s.url
    's3://bucket/hello/world?qwe1=3#ddd'

    >>> s = S3Url("s3://bucket/hello/world#foo?bar=2")
    >>> s.key
    'hello/world#foo?bar=2'
    >>> s.url
    's3://bucket/hello/world#foo?bar=2'
    """

    def __init__(self, url):
        self._parsed = urlparse(url, allow_fragments=False)

    @property
    def bucket(self):
        return self._parsed.netloc

    @property
    def key(self):
        if self._parsed.query:
            return self._parsed.path.lstrip('/') + '?' + self._parsed.query
        else:
            return self._parsed.path.lstrip('/')

    @property
    def url(self):
        return self._parsed.geturl()

session = boto3.Session(profile_name='default')
client = session.client("s3", region_name="us-east-1")

def image_from_s3(url):
    s3_url = S3Url(url)
    resp = client.get_object(Bucket=s3_url.bucket, Key=s3_url.key)
    raw = resp.get('Body').read()
    return Image.open(BytesIO(raw))

import torchvision
import os


class S3Coco(torchvision.datasets.CocoDetection):
    def __init__(self, ann_file):
        super(S3Coco, self).__init__("", ann_file)

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        # return Image.open(os.path.join(self.root, path)).convert("RGB")
        return image_from_s3(path).convert("RGB")

import json
import os
from urllib.parse import urlparse
from utils.split import split_image_with_labels
from PIL import Image

coco_image = {
	"width": int,
	"height": int,
	"id": int,
	"file_name": str
}
coco_annotation = {
	"image_id": int,
	"bbox": list[float],
	"category_id": int,
	"segmentation": list[float],
	"area": float,
	"iscrowd": int,
	"id": int
}
large_coco = json.load(open('datasets/result.json'))
coco_images: list[coco_image] = large_coco['images']
image_id_seq = 0
annotation_id_seq = 0
small_images: list[coco_image] = []
small_annotations: list[coco_annotation] = []


for coco_img in coco_images:
	annotations: list[coco_annotation] = list(filter(lambda ann: ann["image_id"] == coco_img["id"], large_coco['annotations']))
	if len(annotations) == 0:
		continue
	image = image_from_s3(coco_img['file_name'])
	split_images = split_image_with_labels(image=image, labels=annotations,
															  hint_size_min=(800,800), hint_size_max=(1333,1333),
															  overlap=0.1)
	for small in split_images:
		image_id_seq += 1
		image_id = image_id_seq
		left, top, _, _ = small["area"]
		url = urlparse(coco_img['file_name'], allow_fragments=False)
		file_name = url.path.split('/')[-1]
		file_name, file_extension = os.path.splitext(file_name)
		file_name = f"{file_name}_({left},{top}){file_extension}"
		if small['labels'] == []:
			continue
		small["image"].save(f"datasets/small/{file_name}")
		small_images.append({
			"width": small["image"].width,
			"height": small["image"].height,
			"id": image_id,
			"file_name": f"datasets/small/{file_name}"
		})
		for ann in small["labels"]:
			annotation_id_seq += 1
			small_annotations.append({
				"image_id": image_id,
				"bbox": ann["bbox"],
				"category_id": ann["category_id"],
				"segmentation": ann["segmentation"],
				"area": ann["bbox"][2] * ann["bbox"][3],
				"iscrowd": ann["iscrowd"],
				"id": annotation_id_seq
			})

large_coco['images'] = small_images
large_coco['annotations'] = small_annotations
json.dump(large_coco, open('datasets/small.json', 'w'))
