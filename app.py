import cv2
import numpy as np
import os
import sys
import tempfile
import json_tricks as json

# 導入 FastAPI 和相關工具
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# 導入 mmpose 和 mmdet 的核心函式
from mmpose.apis import init_model as init_pose_estimator, inference_topdown
from mmdet.apis import init_detector, inference_detector
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.evaluation.functional import nms

print("--- 正在啟動 FastAPI 應用 (含追蹤邏輯) ---")

# --- 1. 模型初始化 (保持不變) ---
det_config = 'mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
det_checkpoint = 'checkpoints/mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
pose_config = 'mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
pose_checkpoint = 'checkpoints/mmpose/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

try:
    print("[初始化] 正在載入模型...")
    detector = init_detector(det_config, det_checkpoint, device='cuda:0')
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_estimator = init_pose_estimator(pose_config, pose_checkpoint, device='cuda:0')
    print("✅ 所有模型和工具初始化成功！")
except Exception as e:
    print(f"❌ 初始化過程中發生嚴重錯誤: {e}")
    sys.exit()

# --- 2. 建立 FastAPI 應用 (保持不變) ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 輔助函式 ---
def clean_numpy(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)): return int(obj)
    if isinstance(obj, dict): return {k: clean_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list): return [clean_numpy(i) for i in obj]
    return obj

def iou(box1, box2):
    """計算兩個 bounding box 的 IoU (交並比)"""
    x1, y1, x2, y2 = box1[0], box1[1], box1[2], box1[3]
    x3, y3, x4, y4 = box2[0], box2[1], box2[2], box2[3]
    inter_x1, inter_y1 = max(x1, x3), max(y1, y3)
    inter_x2, inter_y2 = min(x2, x4), min(y2, y4)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1, area2 = (x2 - x1) * (y2 - y1), (x4 - x3) * (y4 - y3)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

# --- 3. API 路由 (Endpoint) ---
@app.post("/pose_video")
async def estimate_pose_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        video_path = temp_video_file.name
        temp_video_file.write(await file.read())
    
    background_tasks.add_task(os.remove, video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return JSONResponse(status_code=500, content={"error": "無法開啟影片檔案"})

    frame_idx = 0
    frame_results = []
    
    # 💡【追蹤邏輯】初始化 tracked_bbox，用來記住上一幀的投手位置
    tracked_bbox = None

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            print(f"✅ 正在處理第 {frame_idx} 幀的骨架數據...")

            det_result = inference_detector(detector, frame)
            pred_instance = det_result.pred_instances.cpu().numpy()
            person_bboxes = pred_instance.bboxes[pred_instance.labels == 0]
            person_scores = pred_instance.scores[pred_instance.labels == 0]

            current_bbox = None
            if len(person_bboxes) > 0:
                # 💡【追蹤邏輯】
                if tracked_bbox is None:
                    # 如果是第一幀或之前跟丟了，就找分數最高的人
                    best_bbox_idx = np.argmax(person_scores)
                    current_bbox = person_bboxes[best_bbox_idx]
                    tracked_bbox = current_bbox # 記住這個人的位置
                else:
                    # 如果有追蹤目標，就計算 IoU 找出最像的人
                    ious = [iou(tracked_bbox, box) for box in person_bboxes]
                    best_bbox_idx = np.argmax(ious)
                    # 如果最像的人 IoU > 0.3，就認定是他
                    if ious[best_bbox_idx] > 0.3:
                        current_bbox = person_bboxes[best_bbox_idx]
                        tracked_bbox = current_bbox # 更新追蹤目標的位置
                    else:
                        # 如果 IoU 太低，代表可能跟丟了，下一幀重新找分數最高的
                        tracked_bbox = None 
            
            # 如果這一幀最終沒有找到目標，就記錄空結果
            if current_bbox is None:
                frame_results.append({"frame_idx": frame_idx, "predictions": []})
                frame_idx += 1
                continue

            # 對找到的目標進行姿勢估計
            bboxes_for_pose = [current_bbox]
            pose_results = inference_topdown(pose_estimator, frame, bboxes_for_pose)
            data_samples = merge_data_samples(pose_results)

            # 轉換為 JSON
            predictions = data_samples.get("pred_instances", None)
            split_preds = split_instances(predictions)
            cleaned_preds = clean_numpy(split_preds)

            frame_results.append({
                "frame_idx": frame_idx,
                "predictions": cleaned_preds,
            })
            
            frame_idx += 1
    finally:
        cap.release()
    
    return JSONResponse(content={"frames": frame_results})