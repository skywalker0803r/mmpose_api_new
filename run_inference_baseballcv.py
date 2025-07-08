import cv2
import numpy as np
from baseballcv.detectors import PitcherDetector
from mmpose.apis import init_model as init_pose_estimator, inference_topdown
from mmpose.structures import merge_data_samples, split_instances

def clean_numpy(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)): return int(obj)
    if isinstance(obj, dict): return {k: clean_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list): return [clean_numpy(i) for i in obj]
    return obj

def run_inference_with_baseballcv(video_path: str,
                                   pose_config: str,
                                   pose_checkpoint: str,
                                   device: str = 'cuda:0'):
    # 初始化投手偵測器（可用 yolov5 或 fasterrcnn）
    pitcher_detector = PitcherDetector(model_type='yolov5', device=device)

    # 初始化 RTMPose
    pose_estimator = init_pose_estimator(pose_config, pose_checkpoint, device=device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("無法開啟影片")

    frame_idx = 0
    frame_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 使用 baseballcv 抓出投手 bbox
        pitcher_bboxes = pitcher_detector.detect(frame)

        if not pitcher_bboxes:
            # 沒抓到投手
            frame_results.append({"frame_idx": frame_idx, "predictions": []})
            frame_idx += 1
            continue

        # 預設只抓第一個投手框（通常就一個）
        x1, y1, x2, y2, conf = pitcher_bboxes[0]
        bbox = [x1, y1, x2, y2]

        # 姿勢推論
        pose_results = inference_topdown(pose_estimator, frame, [bbox])
        data_samples = merge_data_samples(pose_results)
        predictions = data_samples.get("pred_instances", None)
        split_preds = split_instances(predictions)
        cleaned_preds = clean_numpy(split_preds)

        frame_results.append({
            "frame_idx": frame_idx,
            "predictions": cleaned_preds[0] if cleaned_preds else [],
        })

        frame_idx += 1

    cap.release()
    return {"frames": frame_results}