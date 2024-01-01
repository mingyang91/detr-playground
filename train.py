import torch
import torchvision
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
from lightning.pytorch import LightningModule, LightningDataModule
from lightning.pytorch.cli import LightningCLI


class CocoDetection(torchvision.datasets.CocoDetection):
  def __init__(self, img_folder, ann_file, processor):
    super(CocoDetection, self).__init__(img_folder, ann_file)
    self.processor = processor

  def __getitem__(self, idx):
    # read in PIL image and target in COCO format
    # feel free to add data augmentation here before passing them to the next step
    img, target = super(CocoDetection, self).__getitem__(idx)

    # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
    image_id = self.ids[idx]
    target = {'image_id': image_id, 'annotations': target}
    encoding = self.processor(
        images=img, annotations=target, return_tensors="pt")
    # remove batch dimension
    pixel_values = encoding["pixel_values"].squeeze()
    target = encoding["labels"]  # remove batch dimension

    return pixel_values, target


ds = CocoDetection(img_folder='.', ann_file='datasets/small.json', processor=None)
id2label = {k: v['name'] for k, v in ds.coco.cats.items()}


class Detr(LightningModule):
  def __init__(self):
    super().__init__()
    self.lr = 5e-5
    self.lr_backbone = 5e-6
    self.weight_decay = 5e-5
    # replace COCO classification head with custom head
    # we specify the "no_timm" variant here to not rely on the timm library
    # for the convolutional backbone
    self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                        revision="no_timm",
                                                        num_labels=len(id2label),
                                                        ignore_mismatched_sizes=True).to(self.device)
    # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896

  def forward(self, pixel_values, pixel_mask):
    outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    return outputs

  def common_step(self, batch, batch_idx):
    pixel_values = batch["pixel_values"]
    pixel_mask = batch["pixel_mask"]
    labels = [{k: v.to(self.device) for k, v in t.items()} 
              for ts in batch["labels"] 
              for t in ts]

    outputs = self.model(pixel_values=pixel_values,
                         pixel_mask=pixel_mask, labels=labels)

    loss = outputs.loss
    loss_dict = outputs.loss_dict

    return loss, loss_dict

  def training_step(self, batch, batch_idx):
    loss, loss_dict = self.common_step(batch, batch_idx)
    # logs metrics for each training_step,
    # and the average across the epoch
    self.log("training_loss", loss)
    for k, v in loss_dict.items():
      self.log("train_" + k, v.item())

    return loss

  def validation_step(self, batch, batch_idx):
    loss, loss_dict = self.common_step(batch, batch_idx)
    self.log("validation_loss", loss)
    for k, v in loss_dict.items():
      self.log("validation_" + k, v.item())

    return loss

  def configure_optimizers(self):
    param_dicts = [
        {
            "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": self.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)

    return optimizer


class DetrDataModel(LightningDataModule):
  def __init__(self):
    super().__init__()
    self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    coco = CocoDetection(
        img_folder='.', ann_file='datasets/small.json', processor=self.processor)
    train_ratio = 0.8
    train_size = int(train_ratio * len(coco))
    val_size = len(coco) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(coco, [train_size, val_size])

    self.train_dataset = train_dataset
    self.val_dataset = val_dataset

  def collate_fn(self, batch):
    pixel_values = [item[0] for item in batch]
    encoding = self.processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch

  def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size=4,
        collate_fn=self.collate_fn,
        shuffle=True,
        num_workers=11,
        multiprocessing_context='fork' if torch.backends.mps.is_available() else None)

  def val_dataloader(self):
    return DataLoader(
        self.val_dataset,
        batch_size=2,
        collate_fn=self.collate_fn,
        num_workers=11,
        persistent_workers=True,
        multiprocessing_context='fork' if torch.backends.mps.is_available() else None)


def cli_main():
  cli = LightningCLI(Detr, DetrDataModel)
  # note: don't call fit!!


if __name__ == "__main__":
  cli_main()
  # note: it is good practice to implement the CLI in a function and call it in the main if block
