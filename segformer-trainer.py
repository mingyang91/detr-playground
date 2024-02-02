import torch
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from PIL import Image
import numpy as np
import cv2
import json
import os

# {'building', 'car', 'crack', 'dog', 'litter', 'pavement', 'truck', 'tyer'}
id2label = {
    0: 'background',
    1: 'building',
    2: 'car',
    3: 'crack',
    4: 'dog',
    5: 'litter',
    6: 'pavement',
    7: 'truck',
    8: 'tyer'
}

label2id = {v: k for k, v in id2label.items()}


def split_mask_into_tiles(mask, tile_size=(512, 512), overlap=0):
    """
    Splits a mask into smaller tiles.

    Args:
        mask (numpy.ndarray): The mask to split, expected to be a 2D array.
        tile_size (tuple): The (height, width) of the tiles.
        overlap (int): The overlap between tiles in pixels.

    Returns:
        A list of mask tiles as numpy.ndarray.
    """
    tiles = []
    h, w = mask.shape
    stride_h, stride_w = tile_size[0] - overlap, tile_size[1] - overlap

    for y in range(0, h, stride_h):
        for x in range(0, w, stride_w):
            tile = mask[y:y + tile_size[0], x:x + tile_size[1]]
            if tile.shape[0] < tile_size[0] or tile.shape[1] < tile_size[1]:
                tile = np.pad(tile, ((0, max(0, tile_size[0] - tile.shape[0])),
                                     (0, max(0, tile_size[1] - tile.shape[1]))),
                              'constant', constant_values=0)  # Assume 0 as the padding value for masks
            tiles.append(tile)
    return tiles


def split_image_into_tiles(image, tile_size=(512, 512), overlap=64):
    """
    Split an image into smaller tiles, optionally with overlap.
    Args:
        image (numpy.ndarray): The image to split.
        tile_size (tuple): The dimensions (height, width) of the tiles.
        overlap (int): The overlap between tiles in pixels.

    Returns:
        list of numpy.ndarray: A list of tile images.
    """
    tiles = []
    h, w = image.shape[:2]
    stride_h, stride_w = tile_size[0] - overlap, tile_size[1] - overlap

    for y in range(0, h, stride_h):
        for x in range(0, w, stride_w):
            tile = image[y:y + tile_size[0], x:x + tile_size[1]]
            # If the tile is smaller than the expected size (at edges), pad it
            if tile.shape[0] < tile_size[0] or tile.shape[1] < tile_size[1]:
                tile = np.pad(tile, (
                    (0, max(0, tile_size[0] - tile.shape[0])), (0, max(0, tile_size[1] - tile.shape[1])), (0, 0)),
                              mode='constant', constant_values=0)
            tiles.append(tile)
    return tiles


def polygons_to_mask(img_shape, shapes):
    """
    Convert polygons to a binary mask.
    Args:
        img_shape: A tuple (height, width) for the shape of the mask.
        shapes: A list of shape dictionaries, each containing 'points' and 'label'.
    Returns:
        A binary mask as a numpy array with shape (height, width).
    """
    mask = np.zeros(img_shape, dtype=np.uint8)
    for shape in shapes:
        if shape['shape_type'] == 'polygon':
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], label2id[shape['label']])  # Fill the polygon with ones
    return mask


def custom_collate_fn(batch):
    # Assuming each element in `batch` is a tuple (image_tiles, mask_tiles)
    # where `image_tiles` and `mask_tiles` are lists of numpy arrays

    # Separate image and mask tiles
    batch_image_tiles = [item[0] for item in batch]  # List of lists of image tiles
    batch_mask_tiles = [item[1] for item in batch]  # List of lists of mask tiles

    # Flatten the lists and convert numpy arrays to tensors
    image_tiles = [tile for sublist in batch_image_tiles for tile in sublist]
    mask_tiles = [tile for sublist in batch_mask_tiles for tile in sublist]

    # Stack the tensors to create batches
    images_stacked = torch.stack(image_tiles)
    masks_stacked = torch.stack(mask_tiles)

    return images_stacked, masks_stacked


class CustomSemanticSegmentationDataset(Dataset):
    def __init__(self, ann_dir, feature_extractor, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            ann_dir (string): Directory with all the annotation JSON files.
            feature_extractor: Feature extractor from the transformers library.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.ann_dir = ann_dir
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.anns = [file for file in os.listdir(ann_dir) if file.endswith(".json")]

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann_path = os.path.join(self.ann_dir, self.anns[idx])
        img_path = os.path.join(self.ann_dir, self.anns[idx].replace('.json', '.jpg'))

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Load annotation and create mask
        with open(ann_path) as f:
            anns = json.load(f)
        mask = polygons_to_mask(image.shape[:2], anns['shapes'])

        tiles = split_image_into_tiles(image)

        # Optionally split the mask here in the same way if training
        masks = split_mask_into_tiles(mask)
        if self.transform:
            transformed = [
                self.transform(image=tile, mask=mask)
                for tile, mask in zip(tiles, masks)
            ]
            tiles = [t['image'] for t in transformed]
            masks = [t['mask'] for t in transformed]
        masks = [torch.as_tensor(mask, dtype=torch.long) for mask in masks]
        tiles = [
            self.feature_extractor(images=t, return_tensors="pt")['pixel_values'].squeeze()
            for t in tiles
        ]
        # Return a list of processed tiles
        # This assumes inference mode. For training, you'll want to process and return masks similarly.
        return tiles, masks


class SegformerFineTuner(LightningModule):
    def __init__(self, num_labels):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b5", num_labels=num_labels)
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b5")

    def forward(self, pixel_values, labels):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch[0], batch[1])
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch[0], batch[1])
        loss = outputs.loss
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer

    def train_dataloader(self):
        train_dataset = CustomSemanticSegmentationDataset(
            "/Users/famer.me/Repository/crack-in-asphalt/2023-12-04/Origin/1", self.feature_extractor)
        return DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

    def val_dataloader(self):
        val_dataset = CustomSemanticSegmentationDataset(
            "/Users/famer.me/Repository/crack-in-asphalt/2023-12-04/Origin/1", self.feature_extractor)
        return DataLoader(val_dataset, batch_size=4, collate_fn=custom_collate_fn)


# Define training


def train_model():
    # Replace YOUR_NUM_LABELS with the actual number of labels in your dataset
    model = SegformerFineTuner(num_labels=len(id2label.keys()))
    trainer = Trainer(max_epochs=10, callbacks=[ModelCheckpoint(monitor='val_loss'), LearningRateMonitor()])
    trainer.fit(model)


if __name__ == '__main__':
    train_model()
