import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from .models import ClassificationLoss, load_model, save_model
# from .utils import load_data
from homework.datasets.road_dataset import load_data

# Define Dice Loss for segmentation
class DiceLoss(nn.Module):
    def __init__(self, num_classes=3, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, target):
        # Apply softmax on logits to get probabilities
        probs = torch.softmax(logits, dim=1)  # Apply softmax across the class dimension

        # Convert target to one-hot encoding
        target_one_hot = torch.nn.functional.one_hot(target.long(), num_classes=logits.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2)  # Change shape to match logits
        # Compute intersection and union
        intersection = torch.sum(probs * target_one_hot, dim=[2, 3])  # Sum over height and width
        union = torch.sum(probs, dim=[2, 3]) + torch.sum(target_one_hot, dim=[2, 3])

        # Compute Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return dice.mean()

# Define IoU for evaluation metrics
def compute_iou(pred, target, num_classes=3):
    iou = []
    for i in range(num_classes):
        # Calculate intersection and union for each class
        intersection = torch.sum((pred == i) & (target == i)).float()
        union = torch.sum((pred == i) | (target == i)).float()
        iou.append(intersection / (union + 1e-6))  # Avoid division by zero
    return torch.mean(torch.tensor(iou))

def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    # Loss functions
    segmentation_loss = DiceLoss() 
    depth_loss = nn.L1Loss()
   
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    global_step = 0
    metrics = {"train_acc": [], "val_acc": [], "train_depth_error": [], "val_depth_error": []}

    # training loop
    for epoch in range(num_epoch):
        model.train()
        
        for data in train_data:
        
            img = data["image"].to(device)     # Move image to device (GPU or CPU)
            label = data["track"].to(device)   # Move track to device
            depth = data["depth"].to(device)   # Move depth to device
             
            # TODO1: implement training step
            # Forward pass
            if model_name == "detector":  # for the detector model
              logits, raw_depth = model(img)
              # Compute losses
              segmentation_loss_value = segmentation_loss(logits, label)
              depth_loss_value = depth_loss(raw_depth.squeeze(1), depth)
              # Total loss
              loss = segmentation_loss_value + depth_loss_value
                
            else:  # for other models like classifier
              logits = model(img)  # Classifier directly returns logits
              loss = loss_func(logits, label)

            # Backward pass and optimization
            optimizer.zero_grad()  # clear previous gradients
            loss.backward()  # compute new gradients
            optimizer.step()  # update weights using gradients

            # Calculate training accuracy
            _, predictions = logits.max(1)  # get the predicted class (index of max logit)
            correct = (predictions == label).sum().item()
            accuracy = correct / label.size(0)
            iou = compute_iou(predictions, label)  # Compute IoU for training
            depth_error = torch.abs(raw_depth - depth).mean().item()
            metrics["train_acc"].append(accuracy)
            metrics["train_depth_error"].append(depth_error)
            global_step += 1

        # disable gradient computation and switch to evaluation mode
            with torch.no_grad():
              model.eval()
            
            for img, track, depth in val_data:
                img = data["image"].to(device)     # Move image to device (GPU or CPU)
                label = data["track"].to(device)   # Move track to device
                depth = data["depth"].to(device)   # Move depth to device
            
                # TODO: compute validation accuracy
                # Forward pass (same as in training, but no backpropagation here)
                if model_name == "detector":  # For the detector model
                    logits, raw_depth = model(img)
            
                # Compute losses
                    segmentation_loss_value = segmentation_loss(logits, label)
                    depth_loss_value = depth_loss(raw_depth.squeeze(1), depth)
            
                    # Total loss
                    loss = segmentation_loss_value + depth_loss_value
            
                else:  # For other models like classifier
                    logits = model(img)  # Classifier directly returns logits
                    loss = loss_func(logits, label)
               # Calculate validation accuracy
                _, preds = logits.max(1)
                correct = (preds == label).sum().item()
                accuracy = correct / label.size(0)
                iou = compute_iou(preds, label)  # Compute IoU for validation
                depth_error = torch.abs(raw_depth - depth).mean().item()

                metrics["val_acc"].append(accuracy)
                metrics["val_iou"].append(iou)
                metrics["val_depth_error"].append(depth_error)


        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()
        epoch_train_iou = torch.as_tensor(metrics["train_iou"]).mean()
        epoch_val_iou = torch.as_tensor(metrics["val_iou"]).mean()
        epoch_train_depth_error = torch.as_tensor(metrics["train_depth_error"]).mean()
        epoch_val_depth_error = torch.as_tensor(metrics["val_depth_error"]).mean()

        logger.add_scalar("train/accuracy", epoch_train_acc, epoch)
        logger.add_scalar("val/accuracy", epoch_val_acc, epoch)
        logger.add_scalar("train/iou", epoch_train_iou, epoch)
        logger.add_scalar("val/iou", epoch_val_iou, epoch)
        logger.add_scalar("train/depth_error", epoch_train_depth_error, epoch)
        logger.add_scalar("val/depth_error", epoch_val_depth_error, epoch)
        
        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epoch}: "
                f"Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}, "
                f"Train IoU: {epoch_train_iou:.4f}, Val IoU: {epoch_val_iou:.4f}, "
                f"Train Depth Error: {epoch_train_depth_error:.4f}, Val Depth Error: {epoch_val_depth_error:.4f}")
            

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args())) 
