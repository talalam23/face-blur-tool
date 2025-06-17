from yolov5facedetector.face_detector import YoloDetector
import cv2
import torch
from huggingface_hub import hf_hub_download
import os
import sys
import numpy as np

original_model = YoloDetector(target_size=640)

def detect_faces_library(image_path, conf_threshold=0.3):
    img = cv2.imread(image_path)
    bboxes, confs, points = original_model.predict(img)
    return bboxes

custom_model = None

def load_custom_model_manually():
    global custom_model
    
    hf_repo_id = "talalam23/yolov5-face-detector-t"
    hf_filename = "yolov5s.pt"
    
    yolov5_source_path = 'yolov5'
    
    if not os.path.exists(yolov5_source_path):
        print(f"FATAL ERROR: The '{yolov5_source_path}' folder was not found.")
        print("Please run 'git clone https://github.com/ultralytics/yolov5' in your terminal.")
        return
        
    sys.path.insert(0, os.path.abspath(yolov5_source_path))
    from models.yolo import Model
    from utils.general import check_yaml
    
    print(f"Downloading '{hf_filename}' from '{hf_repo_id}'...")
    try:
        model_weights_path = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename)
        print(f"Model weights downloaded to: {model_weights_path}")
        
        config_path_relative = os.path.join('models', 'yolov5s.yaml')
        config_path_full = os.path.join(yolov5_source_path, config_path_relative)
        config_path = check_yaml(config_path_full)
        
        model = Model(cfg=config_path, ch=3, nc=1) 
        
        import pathlib
        import platform
        if platform.system() == 'Windows':
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
        try:
            ckpt = torch.load(model_weights_path, map_location='cpu')
        finally:
            if platform.system() == 'Windows':
                pathlib.PosixPath = temp
        
        model.load_state_dict(ckpt['model'].float().state_dict())
        model.eval()
        
        custom_model = model
        print("Custom model loaded successfully using the manual method.")

    except Exception as e:
        print(f"FATAL Error loading custom model manually: {e}")
        import traceback
        traceback.print_exc()
        custom_model = None
    finally:
        sys.path.pop(0)


def detect_faces_custom(image_path, conf_threshold=0.1):
    global custom_model
    if custom_model is None:
        load_custom_model_manually()
        if custom_model is None:
            return []

    img0 = cv2.imread(image_path)
    if img0 is None:
        return []
    img = cv2.resize(img0, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(torch.float32)
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    sys.path.insert(0, os.path.abspath('yolov5'))
    from utils.general import non_max_suppression, scale_boxes
    try:
        pred = custom_model(img)[0]
        pred = non_max_suppression(pred, conf_thres=conf_threshold, iou_thres=0.45)
        det = pred[0]

        bboxes = []
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                bboxes.append([int(c) for c in xyxy])
    finally:
        sys.path.pop(0)

    return [bboxes]