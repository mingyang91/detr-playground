from segformer_trainer import SegformerFineTuner, id2label
from transformers import SegformerImageProcessor
processor = SegformerImageProcessor.from_pretrained('nvidia/mit-b5')
model = SegformerFineTuner.load_from_checkpoint("./lightning_logs/version_41/checkpoints/epoch=99-step=300.ckpt", num_labels=len(id2label.keys()))


model.model.push_to_hub("mingyang91/segformer-for-surveillance")
processor.push_to_hub("mingyang91/segformer-for-surveillance")