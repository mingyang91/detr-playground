import json
import copy
from typing import Optional
from PIL import Image

bbox = [float, float, float, float]

annotation = {
    "id": int,
    "image_id": int,
    "category_id": int,
    "bbox": bbox,
    "ignore": int,
    "iscrowd": int,
    "area": float,
}

small_image = {
    "image": Image,
    "area": bbox
}

def split_image(image: Image,
                hint_size_min: tuple[int, int],
                hint_size_max: tuple[int, int],
                overlap: float = 0.1) -> list[small_image]:
    """
    Given an image and a hint size, split the image into a list of images.
    New images are overlapped with other images by the overlap ratio.
    :param image: The image to split. typically a large image. 1kx1k ~ 10kx10k
    :param hint_size_min: The minimum size of the output image.
    :param hint_size_max: The maximum size of the output image.
    :param overlap: The overlap ratio of the output image.
    :return: A list of images.
    """
    Wi, Hi = image.size
    Wmin, Hmin = hint_size_min
    Wmax, Hmax = hint_size_max
    assert Wmin <= Wmax <= Wi
    assert Hmin <= Hmax <= Hi
    w_search = search(Wi, Wmin, Wmax, overlap)
    h_search = search(Hi, Hmin, Hmax, overlap)
    if w_search is None or h_search is None:
        raise ValueError('The image is too small to split.')
    w_count, output_width, last_output_width, width_overlap = w_search
    h_count, output_height, last_output_height, height_overlap = h_search
    images = []
    for h_index in range(h_count):
        h = h_index * (output_height - height_overlap)
        for w_index in range(w_count):
            w = w_index * (output_width - width_overlap)
            small = {
                "image": image.crop((w, h, w + output_width, h + output_height)),
                "area": (w, h, output_width, output_height)
            }
            images.append(small)
        if last_output_width > 0:
            w = Wi - output_width
            small = {
                "image": image.crop((w, h, w + output_width, h + output_height)),
                "area": (w, h, output_width, output_height)
            }
            images.append(small)
    return images


def search(input: int,
           output_min: int,
           output_max: int,
           overlap: float) -> Optional[tuple[int, int, int, int]]:
    """
    example 1:
    input: 8000, output: 1000, overlap: 0.1
    8000 // (1000 - 100) = 8
    8000 % (1000 - 100) = 800
    count = 8, output = 1000, last_output = 800, overlap_pixels = 100

    example 2:
    input: 7200, output: 800, overlap: 0.1
    7200 // (800 - 80) = 10
    7200 % (800 - 80) = 0
    count = 10, output = 800, last_output = 0, overlap_pixels = 80

    :param input: The length of the input image.
    :param output_min: The minimum length of the output image.
    :param output_max: The maximum length of the output image.
    :param overlap: The overlap ratio of the output image.
    :return: A tuple of (count, output, last_output, overlap_pixels).
    """

    for output in range(output_max, output_min - 1, -1):
        overlap_pixels = int(output * overlap)
        last_output = input % (output - overlap_pixels)
        if last_output == 0 or output_min <= last_output <= output_max:
            count = input // (output - overlap_pixels)
            return count, output, last_output, overlap_pixels

    return None

def box_intersected(box1: bbox, box2: bbox) -> bool:
    """
    Check if two boxes are intersected.
    :param box1: The first box.
    :param box2: The second box.
    :return: True if the two boxes are intersected.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return x1 < x2 + w2 and x2 < x1 + w1 and y1 < y2 + h2 and y2 < y1 + h1

def fit_in_area(annotations: list[annotation], in_area: bbox) -> list[annotation]:
    result = []
    for old in annotations:
        ann = copy.deepcopy(old)
        result.append(ann)
        x, y, w, h = ann["bbox"]
        if x < in_area[0]:
            ann["bbox"][0] = 0
        else:
            ann["bbox"][0] -= in_area[0]
        if y < in_area[1]:
            ann["bbox"][1] = 0
        else:
            ann["bbox"][1] -= in_area[1]
        if x + w > in_area[0] + in_area[2]:
            ann["bbox"][2] = in_area[2] - ann["bbox"][0]
        if y + h > in_area[1] + in_area[3]:
            ann["bbox"][3] = in_area[3] - ann["bbox"][1]
    return result

small_image_with_labels = {
    "image": Image,
    "area": bbox,
    "labels": list[annotation]
}

def split_image_with_labels(image: Image,
                            labels: list[annotation],
                            hint_size_min: tuple[int, int],
                            hint_size_max: tuple[int, int],
                            overlap: float = 0.1) -> list[small_image]:
    small_imgs = split_image(image, hint_size_min, hint_size_max, overlap)
    result = []
    for small_img in small_imgs:
        small_labels = [ann for ann in labels if box_intersected(ann["bbox"], small_img["area"])]
        small_labels = fit_in_area(small_labels, small_img["area"])
        result.append({
            "image": small_img["image"],
            "area": small_img["area"],
            "labels": small_labels
        })
    return result
    

if __name__ == '__main__':
    image = Image.open('../datasets/Das3300161.jpg')
    small_imgs = split_image(image, (800, 800), (1000, 1000), 0.1)
    labels = json.load(open('../datasets/result.json'))
    annotations = list(filter(lambda ann: ann["image_id"] == 28, labels["annotations"]))
    for small_image in small_imgs:
        small_labels = [ann for ann in annotations if box_intersected(ann["bbox"], small_image["area"])]
        small_labels = fit_in_area(small_labels, small_image["area"])
        # save small_labels to json
        json.dump(small_labels, open('datasets/' + str(small_image["area"]) + '.json', 'w'))
        # save small_image["image"] to file
        small_image["image"].save('datasets/' + str(small_image["area"]) + '.jpg')


