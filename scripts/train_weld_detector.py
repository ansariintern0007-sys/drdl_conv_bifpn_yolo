# ============================================
# TRAINING PIPELINE — SAFER REVISED VERSION
# ============================================

import os
import csv
import math
import random

import cv2
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import nms

from scripts.convnext_bifpn_yolo import ConvNeXtBiFPNYOLO


# ================= SETTINGS =================
CLASS_NAMES = [
    "Porosity",
    "Crack",
    "Lack_of_Fusion",
    "Lack_of_Penetration",
    "Slag_Inclusion",
]

NUM_CLASSES = len(CLASS_NAMES)
CLASS_WEIGHTS = torch.tensor([0.2, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)

IMG_SIZE = 640
BATCH_SIZE = 1
EPOCHS = 30
LR = 3e-5
WEIGHT_DECAY = 0.05
SEED = 42

TRAIN_ROOT = "/media/aid-pc/My1TB_2/Swin Yolo Model/dataset"
RUNS_DIR = "runs"

NUM_WORKERS = 2
PIN_MEMORY = True

OBJ_LOSS_WEIGHT = 1.0
CLS_LOSS_WEIGHT = 1.0
BOX_LOSS_WEIGHT = 5.0

CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.50
MATCH_IOU_THRESHOLD = 0.50


# ================= REPRODUCIBILITY =================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================= DATASET =================
class WeldDataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        self.img_dir = os.path.join(root, "images", split)
        self.label_dir = os.path.join(root, "labels", split)

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        self.files = sorted(
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        name = self.files[idx]
        img_path = os.path.join(self.img_dir, name)
        label_path = os.path.join(self.label_dir, name.rsplit(".", 1)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Corrupt image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()

        labels = torch.zeros((0, 5), dtype=torch.float32)

        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            try:
                arr = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)

                # basic cleanup
                arr[:, 0] = np.clip(arr[:, 0], 0, NUM_CLASSES - 1)
                arr[:, 1:] = np.clip(arr[:, 1:], 0.0, 1.0)

                # remove zero/invalid boxes
                valid = (arr[:, 3] > 1e-6) & (arr[:, 4] > 1e-6)
                arr = arr[valid]

                labels = torch.from_numpy(arr).float()
            except Exception:
                labels = torch.zeros((0, 5), dtype=torch.float32)

        return img, labels


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, dim=0), list(targets)


# ================= BOX HELPERS =================
def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    boxes: [N,4] normalized xywh
    returns: [N,4] normalized xyxy
    """
    x, y, w, h = boxes.unbind(dim=1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    out = torch.stack([x1, y1, x2, y2], dim=1)
    return out.clamp(0.0, 1.0)


def bbox_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    box1: [N,4], box2: [M,4] in xyxy
    returns IoU matrix [N,M]
    """
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((box1.shape[0], box2.shape[0]), device=box1.device)

    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = ((box1[:, 2] - box1[:, 0]).clamp(min=0) *
             (box1[:, 3] - box1[:, 1]).clamp(min=0))
    area2 = ((box2[:, 2] - box2[:, 0]).clamp(min=0) *
             (box2[:, 3] - box2[:, 1]).clamp(min=0))

    union = area1[:, None] + area2[None, :] - inter + 1e-7
    return inter / union


# ================= FPN LEVEL ASSIGNMENT =================
def choose_fpn_levels(boxes_xywh: torch.Tensor) -> torch.Tensor:
    """
    boxes_xywh: [N,4] normalized xywh
    returns level ids:
      0 -> P2
      1 -> P3
      2 -> P4
      3 -> P5
    """
    areas = boxes_xywh[:, 2] * boxes_xywh[:, 3]
    levels = torch.empty((len(boxes_xywh),), dtype=torch.long, device=boxes_xywh.device)

    levels[areas < 0.005] = 0
    levels[(areas >= 0.005) & (areas < 0.02)] = 1
    levels[(areas >= 0.02) & (areas < 0.08)] = 2
    levels[areas >= 0.08] = 3

    return levels


# ================= LOSS =================
def compute_loss(obj_preds, cls_preds, reg_preds, targets, device):
    total_loss = torch.tensor(0.0, device=device)
    weights = CLASS_WEIGHTS.to(device)

    valid_images = sum(1 for t in targets if t.numel() > 0)
    if valid_images == 0:
        # still supervise negative objectness everywhere
        for b in range(len(targets)):
            for scale_obj in obj_preds:
                obj_map = scale_obj[b]
                obj_flat = obj_map.permute(1, 2, 0).reshape(-1)
                obj_target = torch.zeros_like(obj_flat)
                total_loss += F.binary_cross_entropy_with_logits(obj_flat, obj_target)
        return total_loss / max(len(targets), 1)

    for i in range(len(targets)):
        t = targets[i]

        # no GTs in this image -> negative objectness only
        if t.numel() == 0:
            for scale_obj in obj_preds:
                obj_map = scale_obj[i]
                obj_flat = obj_map.permute(1, 2, 0).reshape(-1)
                obj_target = torch.zeros_like(obj_flat)
                total_loss += F.binary_cross_entropy_with_logits(obj_flat, obj_target)
            continue

        t_cls = t[:, 0].long()
        t_box = t[:, 1:].clamp(0.0, 1.0)
        assigned_levels = choose_fpn_levels(t_box)

        for level_id, (scale_obj, scale_cls, scale_reg) in enumerate(zip(obj_preds, cls_preds, reg_preds)):
            obj_map = scale_obj[i]   # [1,H,W]
            cls_map = scale_cls[i]   # [C,H,W]
            reg_map = scale_reg[i]   # [4,H,W]

            _, H, W = cls_map.shape

            cls_flat = cls_map.permute(1, 2, 0).reshape(-1, NUM_CLASSES)
            reg_flat = reg_map.permute(1, 2, 0).reshape(-1, 4)
            obj_flat = obj_map.permute(1, 2, 0).reshape(-1)

            obj_target = torch.zeros((H * W,), device=device)

            mask_level = assigned_levels == level_id
            if mask_level.sum() == 0:
                obj_loss = F.binary_cross_entropy_with_logits(obj_flat, obj_target)
                total_loss += OBJ_LOSS_WEIGHT * obj_loss
                continue

            level_cls = t_cls[mask_level]
            level_box = t_box[mask_level]

            cx = (level_box[:, 0] * W).long().clamp(0, W - 1)
            cy = (level_box[:, 1] * H).long().clamp(0, H - 1)
            idx = cy * W + cx

            # remove duplicate target cells on same level by keeping first occurrence
            if idx.numel() > 1:
                idx_unique, first_pos = torch.unique(idx, sorted=False, return_inverse=False, return_counts=False), []
                seen = set()
                for k, v in enumerate(idx.tolist()):
                    if v not in seen:
                        seen.add(v)
                        first_pos.append(k)
                first_pos = torch.tensor(first_pos, device=device, dtype=torch.long)

                idx = idx[first_pos]
                level_cls = level_cls[first_pos]
                level_box = level_box[first_pos]

            obj_target[idx] = 1.0

            pred_cls = cls_flat[idx]                 # [P,C]
            pred_reg = reg_flat[idx].sigmoid()       # [P,4]

            obj_loss = F.binary_cross_entropy_with_logits(obj_flat, obj_target)
            cls_loss = F.cross_entropy(pred_cls, level_cls, weight=weights)
            box_loss = F.smooth_l1_loss(pred_reg, level_box)

            total_loss += (
                OBJ_LOSS_WEIGHT * obj_loss +
                CLS_LOSS_WEIGHT * cls_loss +
                BOX_LOSS_WEIGHT * box_loss
            )

    return total_loss / max(len(targets), 1)


# ================= DECODE PREDICTIONS =================
def decode_predictions_single(obj_preds, cls_preds, reg_preds, img_index: int, device):
    """
    Decodes predictions for one image across all scales.
    Returns:
      boxes_xyxy [N,4]
      scores [N]
      labels [N]
    """
    all_boxes = []
    all_scores = []
    all_labels = []

    for scale_obj, scale_cls, scale_reg in zip(obj_preds, cls_preds, reg_preds):
        obj_map = torch.sigmoid(scale_obj[img_index])   # [1,H,W]
        cls_map = torch.sigmoid(scale_cls[img_index])   # [C,H,W]
        reg_map = torch.sigmoid(scale_reg[img_index])   # [4,H,W]

        obj_flat = obj_map.permute(1, 2, 0).reshape(-1)                    # [HW]
        cls_flat = cls_map.permute(1, 2, 0).reshape(-1, NUM_CLASSES)       # [HW,C]
        reg_flat = reg_map.permute(1, 2, 0).reshape(-1, 4).clamp(0.0, 1.0) # [HW,4]

        cls_scores, cls_labels = cls_flat.max(dim=1)
        scores = obj_flat * cls_scores

        keep = scores > CONF_THRESHOLD
        if keep.sum() == 0:
            continue

        boxes_xyxy = xywh_to_xyxy(reg_flat[keep])
        scores = scores[keep]
        labels = cls_labels[keep]

        all_boxes.append(boxes_xyxy)
        all_scores.append(scores)
        all_labels.append(labels)

    if len(all_boxes) == 0:
        return (
            torch.empty((0, 4), device=device),
            torch.empty((0,), device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # class-aware NMS
    keep_indices = []
    for c in range(NUM_CLASSES):
        cls_mask = labels == c
        if cls_mask.sum() == 0:
            continue
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_keep = nms(cls_boxes, cls_scores, NMS_IOU_THRESHOLD)
        original_idx = torch.where(cls_mask)[0][cls_keep]
        keep_indices.append(original_idx)

    if len(keep_indices) == 0:
        return (
            torch.empty((0, 4), device=device),
            torch.empty((0,), device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )

    keep_indices = torch.cat(keep_indices, dim=0)
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]


# ================= METRICS =================
def compute_metrics(obj_preds, cls_preds, reg_preds, targets, device):
    tp, fp, fn = 0, 0, 0

    for i in range(len(targets)):
        t = targets[i]

        pred_boxes, pred_scores, pred_labels = decode_predictions_single(
            obj_preds, cls_preds, reg_preds, i, device
        )

        if t.numel() == 0:
            fp += int(pred_boxes.shape[0])
            continue

        gt_labels = t[:, 0].long()
        gt_boxes_xyxy = xywh_to_xyxy(t[:, 1:].clamp(0.0, 1.0))

        if pred_boxes.shape[0] == 0:
            fn += int(gt_boxes_xyxy.shape[0])
            continue

        matched_gt = torch.zeros((gt_boxes_xyxy.shape[0],), dtype=torch.bool, device=device)

        # match high-score preds first
        order = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]

        for p_box, p_label in zip(pred_boxes, pred_labels):
            same_class = gt_labels == p_label
            available = same_class & (~matched_gt)

            if available.sum() == 0:
                fp += 1
                continue

            candidate_gt = gt_boxes_xyxy[available]
            ious = bbox_iou_xyxy(p_box.unsqueeze(0), candidate_gt).squeeze(0)

            best_iou, best_local_idx = ious.max(dim=0)
            if best_iou >= MATCH_IOU_THRESHOLD:
                global_indices = torch.where(available)[0]
                matched_gt[global_indices[best_local_idx]] = True
                tp += 1
            else:
                fp += 1

        fn += int((~matched_gt).sum().item())

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    accuracy = tp / (tp + fp + fn + 1e-7)

    return precision, recall, accuracy


# ================= TRAIN =================
def train():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = WeldDataset(TRAIN_ROOT, "train")
    val_dataset = WeldDataset(TRAIN_ROOT, "val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
        drop_last=False,
    )

    model = ConvNeXtBiFPNYOLO(num_classes=NUM_CLASSES).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    os.makedirs(RUNS_DIR, exist_ok=True)
    log_path = os.path.join(RUNS_DIR, "log.csv")

    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "precision", "recall", "accuracy"])

        best_val_loss = float("inf")

        for ep in range(EPOCHS):
            # ===== TRAIN =====
            model.train()
            running_train_loss = 0.0
            train_steps = 0

            pbar = tqdm(train_loader, desc=f"Epoch {ep + 1}/{EPOCHS}")
            for imgs, targets in pbar:
                imgs = imgs.to(device, non_blocking=True)
                targets = [t.to(device) for t in targets]

                optimizer.zero_grad(set_to_none=True)

                try:
                    obj_preds, cls_preds, reg_preds = model(imgs)
                    loss = compute_loss(obj_preds, cls_preds, reg_preds, targets, device)

                    if not torch.isfinite(loss):
                        print("Skipping non-finite loss batch.")
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    running_train_loss += loss.item()
                    train_steps += 1

                    pbar.set_postfix(loss=f"{loss.item():.4f}")

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("CUDA OOM: skipping batch.")
                        optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise

            scheduler.step()
            avg_train_loss = running_train_loss / max(train_steps, 1)

            # ===== VALIDATION =====
            model.eval()
            running_val_loss = 0.0
            total_p, total_r, total_a = 0.0, 0.0, 0.0
            val_steps = 0

            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    targets = [t.to(device) for t in targets]

                    try:
                        obj_preds, cls_preds, reg_preds = model(imgs)
                        loss = compute_loss(obj_preds, cls_preds, reg_preds, targets, device)

                        if not torch.isfinite(loss):
                            continue

                        p, r, a = compute_metrics(obj_preds, cls_preds, reg_preds, targets, device)

                        running_val_loss += loss.item()
                        total_p += p
                        total_r += r
                        total_a += a
                        val_steps += 1

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print("CUDA OOM during validation batch: skipping.")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        raise

            avg_val_loss = running_val_loss / max(val_steps, 1)
            avg_p = total_p / max(val_steps, 1)
            avg_r = total_r / max(val_steps, 1)
            avg_a = total_a / max(val_steps, 1)

            print(f"\nEpoch {ep + 1}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss:   {avg_val_loss:.4f}")
            print(f"Precision:  {avg_p:.4f}")
            print(f"Recall:     {avg_r:.4f}")
            print(f"Accuracy:   {avg_a:.4f}")

            writer.writerow([ep + 1, avg_train_loss, avg_val_loss, avg_p, avg_r, avg_a])
            log_file.flush()

            torch.save(model.state_dict(), os.path.join(RUNS_DIR, "last.pt"))

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(RUNS_DIR, "best.pt"))

    print("\nTraining complete.")
    print(f"Logs saved to: {log_path}")
    print(f"Best model: {os.path.join(RUNS_DIR, 'best.pt')}")
    print(f"Last model: {os.path.join(RUNS_DIR, 'last.pt')}")


if __name__ == "__main__":
    train()