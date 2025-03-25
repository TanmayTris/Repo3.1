import argparse
from datetime import datetime
from pathlib importPath

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb
from torchvision import transforms
from .models import ClassificationLoss, load_model, save_model
from homework.datasets.road_dataset import load_data

def dice_loss(pred, target, smooth=1e-6):
    """Computes the Dice loss for segmentation."""
    pred = torch.softmax(pred, dim=1)
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    return 1 - (2. * intersection + smooth) / (union + smooth)

def focal_loss(pred, target, alpha=0.25, gamma=2.0, smooth=1e-6):
    """Computes the Focal Loss for segmentation."""
    pred = torch.softmax(pred, dim=1)
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    logpt = -nn.CrossEntropyLoss(reduction='none')(pred, target)
    pt = torch.exp(logpt)
    focal_loss = -((1 - pt) ** gamma) * logpt
    return focal_loss.mean()

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)
    
    model = load_model(model_name, **kwargs).to(device)
    model.train()

    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=4, transform=data_transforms)
    val_data = load_data("drive_data/val", shuffle=False)
    
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    else:
        class_weights = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32, device=device)
    
    segmentation_loss = nn.CrossEntropyLoss(weight=class_weights)
    depth_loss = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_data), epochs=num_epoch)
    
    for epoch in range(num_epoch):
        model.train()
        total_iou, total_depth_error, total_tp_depth_error = 0, 0, 0
        count = 0

        for data in train_data:
            img, label, depth = data["image"].to(device), data["track"].to(device), data["depth"].to(device)
            optimizer.zero_grad()
            
            logits, raw_depth = model(img)
            
            seg_loss = segmentation_loss(logits, label) + dice_loss(logits, label).mean() + focal_loss(logits, label).mean()
            depth_loss_value = depth_loss(raw_depth.squeeze(1), depth)
            loss = seg_loss + depth_loss_value
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            _, preds = logits.max(1)
            intersection = (preds & label).float().sum((1, 2))
            union = (preds | label).float().sum((1, 2))
            iou = (intersection / (union + 1e-6)).mean()
            
            depth_error = torch.abs(raw_depth - depth).mean()
            tp_depth_error = torch.abs(raw_depth[label == 1] - depth[label == 1]).mean()
            
            total_iou += iou.item()
            total_depth_error += depth_error.item()
            total_tp_depth_error += tp_depth_error.item()
            count += 1
        
        avg_iou = total_iou / count
        avg_depth_error = total_depth_error / count
        avg_tp_depth_error = total_tp_depth_error / count
        
        logger.add_scalar("train/iou", avg_iou, epoch)
        logger.add_scalar("train/depth_error", avg_depth_error, epoch)
        logger.add_scalar("train/tp_depth_error", avg_tp_depth_error, epoch)
        
        model.eval()
        with torch.no_grad():
            val_iou, val_depth_error, val_tp_depth_error, val_count = 0, 0, 0, 0
            for data in val_data:
                img, label, depth = data["image"].to(device), data["track"].to(device), data["depth"].to(device)
                logits, raw_depth = model(img)
                
                _, preds = logits.max(1)
                intersection = (preds & label).float().sum((1, 2))
                union = (preds | label).float().sum((1, 2))
                val_iou += (intersection / (union + 1e-6)).mean().item()
                
                depth_error = torch.abs(raw_depth - depth).mean()
                tp_depth_error = torch.abs(raw_depth[label == 1] - depth[label == 1]).mean()
                
                val_depth_error += depth_error.item()
                val_tp_depth_error += tp_depth_error.item()
                val_count += 1
            
            avg_val_iou = val_iou / val_count
            avg_val_depth_error = val_depth_error / val_count
            avg_val_tp_depth_error = val_tp_depth_error / val_count
        
        logger.add_scalar("val/iou", avg_val_iou, epoch)
        logger.add_scalar("val/depth_error", avg_val_depth_error, epoch)
        logger.add_scalar("val/tp_depth_error", avg_val_tp_depth_error, epoch)
        
        print(f"Epoch {epoch+1}/{num_epoch}: IoU {avg_iou:.4f}, Depth Error {avg_depth_error:.4f}, TP Depth Error {avg_tp_depth_error:.4f}")
    
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
