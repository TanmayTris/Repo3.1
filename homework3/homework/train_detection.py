import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb
from torchvision import transforms
from .models import ClassificationLoss, load_model, save_model
from homework.datasets.road_dataset import load_data # load

def iou_metric(pred, target, num_classes):
    iou = []
    pred = torch.argmax(pred, dim=1)
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = torch.sum(pred_cls & target_cls)
        union = torch.sum(pred_cls | target_cls)
        if union == 0:
            iou.append(float('nan'))  # Avoid division by zero
        else:
            iou.append(intersection.item() / union.item())
    return np.nanmean(iou)

def custom_depth_loss(pred, target, mask):
    abs_error = torch.abs(pred - target)
    tp_error = abs_error[mask].mean()
    return abs_error.mean(), tp_error

def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    class_weights: list = None,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor()
    ])

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2, transform=data_transforms)
    val_data = load_data("drive_data/val", shuffle=False)

    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    else:
        class_weights = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32, device=device)

    segmentation_loss = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    for epoch in range(num_epoch):
        model.train()
        train_metrics = {"iou": [], "abs_depth_error": [], "tp_depth_error": []}

        for data in train_data:
            img = data["image"].to(device)
            label = data["track"].to(device)
            depth = data["depth"].to(device)

            if model_name == "detector":
                logits, raw_depth = model(img)
                seg_loss = segmentation_loss(logits, label)
                abs_depth_error, tp_depth_error = custom_depth_loss(raw_depth.squeeze(1), depth, label > 0)
                loss = seg_loss + abs_depth_error + tp_depth_error
            else:
                logits = model(img)
                loss = loss_func(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_metrics["iou"].append(iou_metric(logits, label, num_classes=3))
            train_metrics["abs_depth_error"].append(abs_depth_error.item())
            train_metrics["tp_depth_error"].append(tp_depth_error.item())

            global_step += 1

        model.eval()
        val_metrics = {"iou": [], "abs_depth_error": [], "tp_depth_error": []}
        with torch.no_grad():
            for data in val_data:
                img = data["image"].to(device)
                label = data["track"].to(device)
                depth = data["depth"].to(device)

                if model_name == "detector":
                    logits, raw_depth = model(img)
                    abs_depth_error, tp_depth_error = custom_depth_loss(raw_depth.squeeze(1), depth, label > 0)
                else:
                    logits = model(img)

                val_metrics["iou"].append(iou_metric(logits, label, num_classes=3))
                val_metrics["abs_depth_error"].append(abs_depth_error.item())
                val_metrics["tp_depth_error"].append(tp_depth_error.item())

        epoch_train_iou = np.mean(train_metrics["iou"])
        epoch_val_iou = np.mean(val_metrics["iou"])
        epoch_train_abs_depth_error = np.mean(train_metrics["abs_depth_error"])
        epoch_val_abs_depth_error = np.mean(val_metrics["abs_depth_error"])
        epoch_train_tp_depth_error = np.mean(train_metrics["tp_depth_error"])
        epoch_val_tp_depth_error = np.mean(val_metrics["tp_depth_error"])

        logger.add_scalar("train/iou", epoch_train_iou, epoch)
        logger.add_scalar("val/iou", epoch_val_iou, epoch)
        logger.add_scalar("train/abs_depth_error", epoch_train_abs_depth_error, epoch)
        logger.add_scalar("val/abs_depth_error", epoch_val_abs_depth_error, epoch)
        logger.add_scalar("train/tp_depth_error", epoch_train_tp_depth_error, epoch)
        logger.add_scalar("val/tp_depth_error", epoch_val_tp_depth_error, epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epoch}: "
                  f"Train IoU: {epoch_train_iou:.4f}, Val IoU: {epoch_val_iou:.4f}, "
                  f"Train Abs Depth Error: {epoch_train_abs_depth_error:.4f}, Val Abs Depth Error: {epoch_val_abs_depth_error:.4f}, "
                  f"Train TP Depth Error: {epoch_train_tp_depth_error:.4f}, Val TP Depth Error: {epoch_val_tp_depth_error:.4f}")

    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--class_weights", type=float, nargs="*", default=None)

    train(**vars(parser.parse_args()))
