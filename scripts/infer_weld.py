import cv2
import torch
import numpy as np
import torchvision.ops as ops
from scripts.convnext_bifpn_yolo import ConvNeXtBiFPNYOLO


# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "checkpoint.pt"
IMG_SIZE = 640
CONF_THRESH = 0.3
IOU_THRESH = 0.5

CLASS_NAMES = [
    "Porosity",
    "Crack",
    "Lack_of_Fusion",
    "Lack_of_Penetration",
    "Slag_Inclusion"
]


# -------------------------
# LOAD MODEL
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvNeXtBiFPNYOLO(num_classes=5).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# -------------------------
# PREPROCESS
# -------------------------
def preprocess(img):

    orig = img.copy()

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img[:, :, ::-1].astype(np.float32) / 255.0

    mean = np.array([0.485,0.456,0.406],dtype=np.float32)
    std  = np.array([0.229,0.224,0.225],dtype=np.float32)

    img = (img - mean) / std

    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)

    return img.to(device), orig


# -------------------------
# DECODE OUTPUT
# -------------------------
def decode(cls_preds, reg_preds):

    boxes = []
    scores = []
    labels = []

    for cls_p, reg_p in zip(cls_preds, reg_preds):

        B, C, H, W = cls_p.shape

        cls_p = torch.sigmoid(cls_p).view(C, -1).permute(1,0)
        reg_p = reg_p.view(4, -1).permute(1,0)

        conf, cls_ids = cls_p.max(dim=1)

        mask = conf > CONF_THRESH

        if mask.sum() == 0:
            continue

        conf = conf[mask]
        cls_ids = cls_ids[mask]
        reg_p = reg_p[mask]

        # convert xywh → xyxy
        x = reg_p[:,0]
        y = reg_p[:,1]
        w = reg_p[:,2]
        h = reg_p[:,3]

        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2

        box = torch.stack([x1,y1,x2,y2], dim=1)

        boxes.append(box)
        scores.append(conf)
        labels.append(cls_ids)

    if len(boxes) == 0:
        return None, None, None

    boxes = torch.cat(boxes)
    scores = torch.cat(scores)
    labels = torch.cat(labels)

    # NMS
    keep = ops.nms(boxes, scores, IOU_THRESH)

    return boxes[keep], scores[keep], labels[keep]


# -------------------------
# DRAW BOXES
# -------------------------
def draw(img, boxes, scores, labels):

    h, w, _ = img.shape

    for box, score, cls in zip(boxes, scores, labels):

        x1,y1,x2,y2 = box

        # scale back
        x1 = int(x1.item() * w)
        y1 = int(y1.item() * h)
        x2 = int(x2.item() * w)
        y2 = int(y2.item() * h)

        name = CLASS_NAMES[int(cls.item())]

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(
            img,
            f"{name} {score:.2f}",
            (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            1
        )

    return img


# -------------------------
# MAIN
# -------------------------
def run(image_path):

    img = cv2.imread(image_path)

    inp, orig = preprocess(img)

    with torch.no_grad():
        cls_preds, reg_preds = model(inp)

    boxes, scores, labels = decode(cls_preds, reg_preds)

    if boxes is None:
        print("No detections")
        return

    result = draw(orig, boxes, scores, labels)

    cv2.imwrite("output.jpg", result)

    print("Saved → output.jpg")


if __name__ == "__main__":
    run("test.jpg")