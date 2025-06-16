import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import time
import os

log_dir = "runs/tensorboard_logs"
csv_path = "runs/train/face_mask_detector/results.csv"

writer = SummaryWriter(log_dir)

print("📊 Waiting for results.csv to appear...")

while not os.path.exists(csv_path):
    print("⏳ Waiting for results.csv to appear...")
    time.sleep(2)

df = pd.read_csv(csv_path)

for i, row in df.iterrows():
    writer.add_scalar("Loss/train_box_loss", row["train/box_loss"], i)
    writer.add_scalar("Loss/train_cls_loss", row["train/cls_loss"], i)
    writer.add_scalar("Loss/train_dfl_loss", row["train/dfl_loss"], i)
    
    writer.add_scalar("Val/val_box_loss", row["val/box_loss"], i)
    writer.add_scalar("Val/val_cls_loss", row["val/cls_loss"], i)
    writer.add_scalar("Val/val_dfl_loss", row["val/dfl_loss"], i)
    
    writer.add_scalar("Metrics/precision", row["metrics/precision(B)"], i)
    writer.add_scalar("Metrics/recall", row["metrics/recall(B)"], i)
    writer.add_scalar("Metrics/mAP50", row["metrics/mAP50(B)"], i)
    writer.add_scalar("Metrics/mAP50_95", row["metrics/mAP50-95(B)"], i)

writer.close()
print("✅ Logging complete. Run `tensorboard --logdir=runs/tensorboard_logs` to view.")