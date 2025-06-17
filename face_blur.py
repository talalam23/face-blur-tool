import cv2
import os
from yolo_face_detector import detect_faces_library, detect_faces_custom

# Custom model trained and loaded from Hugging Face.
#detect_faces = detect_faces_custom

# Yolov5-face-detector Library model
detect_faces = detect_faces_library

def apply_pixelate(face):
    h, w = face.shape[:2]
    if h == 0 or w == 0:
        return face
    
    pixel_size = 8 
    
    temp = cv2.resize(face, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_gaussian_blur(face):
    if face.size == 0:
        return face
        
    kernel_size = 151
    
    return cv2.GaussianBlur(face, (kernel_size, kernel_size), 30)

def apply_black_box(face):
    face[:] = 0
    return face


def resize_to_multiple_of_32(frame):
    height, width = frame.shape[:2]
    new_height = (height // 32) * 32
    new_width = (width // 32) * 32
    if new_height == 0 or new_width == 0:
        return frame
    if new_height == height and new_width == width:
        return frame
    return cv2.resize(frame, (new_width, new_height))


def blur_faces_image(image_path, blur_type='blur'):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None

    boxes_data = detect_faces(image_path)

    if boxes_data and len(boxes_data) > 0 and boxes_data[0] is not None:
        boxes = boxes_data[0]
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if y2 <= y1 or x2 <= x1: continue
            face = img[y1:y2, x1:x2]
            if face.size == 0: continue

            if blur_type == 'blackbox':
                img[y1:y2, x1:x2] = apply_black_box(face)
            elif blur_type == 'pixelate':
                img[y1:y2, x1:x2] = apply_pixelate(face)
            else:
                img[y1:y2, x1:x2] = apply_gaussian_blur(face)

    output_path = 'blurred_image.jpg'
    cv2.imwrite(output_path, img)
    return output_path


def blur_faces_video(video_path, blur_type='pixelate'):
    cap = cv2.VideoCapture(video_path)
    out = None
    temp_image_path = 'temp_frame_blur.jpg'

    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        if not ret:
            return None

        initial_frame = resize_to_multiple_of_32(frame)
        height, width, _ = initial_frame.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        output_path = 'blurred_video.mp4'
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            return None

        while True:
            frame = cv2.resize(frame, (width, height))
            cv2.imwrite(temp_image_path, frame)
            
            boxes_data = detect_faces(temp_image_path)
            
            if boxes_data and len(boxes_data) > 0 and boxes_data[0] is not None:
                face_boxes = boxes_data[0]
                for box in face_boxes:
                    if not isinstance(box, (list, tuple)) or len(box) != 4: continue
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width - 1, x2), min(height - 1, y2)
                    if y2 <= y1 or x2 <= x1: continue
                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size == 0: continue

                    if blur_type == 'pixelate':
                        frame[y1:y2, x1:x2] = apply_pixelate(face_roi)
                    elif blur_type == 'blackbox':
                        frame[y1:y2, x1:x2] = apply_black_box(face_roi)
                    else:
                        frame[y1:y2, x1:x2] = apply_gaussian_blur(face_roi)

            out.write(frame)
            ret, frame = cap.read()
            if not ret:
                break
        return output_path
    finally:
        if cap and cap.isOpened(): cap.release()
        if out and out.isOpened(): out.release()
        if os.path.exists(temp_image_path): os.remove(temp_image_path)