import os
import sys
import time
import json
import cv2
import torch
import argparse
import numpy as np
from ultralytics import YOLO

sys.path.append(os.getcwd())
from lib.config import config, update_config
from lib.models import hrnet
from lib.utils.utils import get_final_preds
from lib.utils.transforms import get_affine_transform

CFG = 'experiments/300w/face_alignment_300w_hrnet_w18.yaml'
HRNET_WEIGHTS = 'hrnetv2_pretrained/HR18-300W.pth'
YOLO_WEIGHTS = 'yolov8-face/weights/yolov11l-face.pt'
INPUT_DIR = 'images'
OUT_DIR = 'out_landmarks'
SIZE = (256, 256)
CONF_THRESH = 0.7

os.makedirs(OUT_DIR, exist_ok=True)

def load_models(device):
    """Load mô hình YOLO và HRNet với trọng số, chuyển sang eval mode."""
    yolo = YOLO(YOLO_WEIGHTS)
    model = hrnet.get_face_alignment_net(config)
    state = torch.load(HRNET_WEIGHTS, map_location='cpu')
    state = {k.replace('module.', ''): v for k, v in state.get('state_dict', state).items()}
    model.load_state_dict(state, strict=False)
    return yolo, model.to(device).eval()

def preprocess(img, box):
    """Cắt và chuẩn hóa vùng khuôn mặt từ bounding box."""
    x1, y1, x2, y2 = map(int, box)
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
    scale_val = max(x2 - x1, y2 - y1) / 200 * 1.25
    scale = np.array([scale_val, scale_val], dtype=np.float32)
    trans = get_affine_transform(center, scale, 0, SIZE)
    crop = cv2.warpAffine(img, trans, SIZE, flags=cv2.INTER_LINEAR)
    if crop is None:
        return None, None, None
    crop = (crop[..., ::-1] / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, center, scale

def process_image(path, yolo, model, device, collect_results=None):
    """Xử lý ảnh: phát hiện mặt, dự đoán landmark, lưu kết quả hình và JSON."""
    img = cv2.imread(path)
    if img is None:
        print(f"Lỗi ảnh: {path}")
        return

    res = yolo(img)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes else []
    scores = res.boxes.conf.cpu().numpy() if res.boxes else []

    if len(boxes) == 0:
        print(f"Không mặt: {os.path.basename(path)}")
        return

    total = 0
    t0 = time.time()
    landmark_data = []

    for box, conf in zip(boxes, scores):
        if conf < CONF_THRESH:
            continue

        tensor, center, scale = preprocess(img, box)
        if tensor is None:
            continue

        with torch.no_grad():
            out = model(tensor.to(device))

        preds, _ = get_final_preds(out.cpu(), torch.tensor([center]), torch.tensor([scale]))
        landmarks = preds[0].tolist()

        for x, y in landmarks:
            cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
            total += 1

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        landmark_data.append({
            "box": [x1, y1, x2, y2],
            "score": float(conf),
            "landmarks": landmarks
        })

    elapsed = time.time() - t0
    filename = os.path.basename(path)
    out_path = os.path.join(OUT_DIR, filename)
    cv2.putText(img, f"{total} pts | {elapsed:.2f}s", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(out_path, img)

    if collect_results is not None:
        collect_results.append({
            "image": filename,
            "faces": landmark_data
        })
    print(f"{filename} ({total} điểm, {elapsed:.2f}s)")

def main():
    """Hàm chính xử lý toàn bộ ảnh và xuất JSON."""
    update_config(config, argparse.Namespace(cfg=CFG))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo, model = load_models(device)
    imgs = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not imgs:
        print("Không tìm thấy ảnh")
        return

    all_results = []
    for f in imgs:
        process_image(os.path.join(INPUT_DIR, f), yolo, model, device, collect_results=all_results)
    json_path = os.path.join(OUT_DIR, 'all_results.json')
    with open(json_path, 'w') as jf:
        json.dump(all_results, jf, indent=2)

if __name__ == '__main__':
    main()
