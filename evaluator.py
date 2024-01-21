from typing import Callable

from PIL.Image import Image
from coco_eval import CocoEvaluator
from pycocotools.coco import COCO
from tqdm import tqdm

from yolo_dataset import YoloDataset
from yolo_fire import model

image_loader = Callable[[str], Image]
infer_fn = Callable[[Image, int, float], list]

def infer(image: Image, image_id: int, confidence_threshold):
    results = model(source=image)
    return [
        ann
        for result in results
        for ann in yolo_boxes_to_coco_annotations(image_id, 
                                                  result.boxes,
                                                  confidence_threshold=confidence_threshold)
    ]


def evaluate(coco_gt: COCO, infer_fn: infer_fn, loader: image_loader, confidence_threshold=0.6):
    # initialize evaluator with ground truth (gt)
    evaluator = CocoEvaluator(coco_gt=coco_gt, iou_types=["bbox"])

    print("Running evaluation...")
    for image_id, annotations in tqdm(coco_gt.imgToAnns.items()):
        # get the inputs
        image = coco_gt.imgs[image_id]
        image = loader(image["file_name"])
        coco_anns = infer_fn(image=image, image_id=image_id, confidence_threshold=confidence_threshold)
        if len(coco_anns) == 0:
            continue
        evaluator.update(coco_anns)

    if len(evaluator.eval_imgs["bbox"]) == 0:
        print("No detections!")
        return
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()


def yolo_boxes_to_coco_annotations(image_id: int, yolo_boxes, confidence_threshold=0.6):
    return [
        {
            "image_id": image_id,
            "category_id": box.cls.tolist()[0],
            "area": box.xywh.tolist()[0][2] * box.xywh.tolist()[0][3],
            "bbox": box.xywh.tolist()[0],
            "score": box.conf.tolist()[0],
        }
        for box in yolo_boxes if box.conf.tolist()[0] > confidence_threshold
    ]


if __name__ == '__main__':
    yolo_dataset = YoloDataset.from_zip_file('tests/coco8.zip')
    coco_gt = yolo_dataset.to_coco()
    evaluate(coco_gt=coco_gt, loader=yolo_dataset.load_image, confidence_threshold=0.1)
