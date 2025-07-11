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

# 嘗試導入 ffmpegcv 以實現 GPU 影片解碼
try:
    import ffmpegcv
    print("✅ 成功導入 ffmpegcv，將嘗試使用 GPU 加速影片解碼。")
    USE_FFMPEGCV = True
except ImportError:
    print("❌ 未能導入 ffmpegcv，將使用 OpenCV 進行影片解碼。請確保已安裝 ffmpegcv (pip install ffmpegcv) 並配置好相關環境以獲得最佳性能。")
    USE_FFMPEGCV = False

print("--- 正在啟動 FastAPI 應用 (含追蹤邏輯與加速優化) ---")

# --- 1. 模型初始化 ---
det_config = 'mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
det_checkpoint = 'checkpoints/mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
pose_config = 'mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
pose_checkpoint = 'checkpoints/mmpose/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

try:
    print("[初始化] 正在載入模型...")
    detector = init_detector(det_config, det_checkpoint, device='cuda:0')
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    pose_estimator = init_pose_estimator(pose_config, pose_checkpoint, device='cuda:0')

    # 【加速優化點 1】禁用 flip_test - 已有，非常有效
    if hasattr(pose_estimator.cfg.model, 'test_cfg') and 'flip_test' in pose_estimator.cfg.model.test_cfg:
        pose_estimator.cfg.model.test_cfg.flip_test = False
        print("✅ 已禁用 pose_estimator 的 flip_test 以加速推論。")
    
    print("✅ 所有模型和工具初始化成功！")
except Exception as e:
    print(f"❌ 初始化過程中發生嚴重錯誤: {e}")
    sys.exit(1)

# --- 2. 建立 FastAPI 應用 ---
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
    """
    遞歸地將 NumPy 類型轉換為標準 Python 類型，以便 JSON 序列化。
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, float)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: clean_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_numpy(item) for item in obj] # Changed 'i' to 'item' here
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

    # 【加速優化點 2】嘗試使用 ffmpegcv 進行 GPU 影片解碼 - 已有，非常有效
    if USE_FFMPEGCV:
        try:
            cap = ffmpegcv.VideoCapture(video_path, gpu=0)
            print(f"✅ 成功使用 ffmpegcv 載入影片: {video_path}")
        except Exception as e:
            print(f"嘗試使用 ffmpegcv 載入影片失敗，退回使用 OpenCV: {e}")
            cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return JSONResponse(status_code=500, content={"error": "無法開啟影片檔案"})

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_results = [None] * total_frames if total_frames > 0 else []

    frame_idx = 0
    
    tracked_bbox = None
    last_processed_predictions = [] # Store the last successful predictions

    # 【新加速優化點】跳幀處理設置
    frame_skip_interval = 3 # 處理每2幀，即跳過一幀。可以根據需求調整此值。

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            current_frame_predictions = []

            # 只在指定的間隔進行完整的檢測和姿態估計
            if frame_idx % frame_skip_interval == 0:
                det_result = inference_detector(detector, frame)
                pred_instance = det_result.pred_instances.cpu().numpy()
                person_bboxes = pred_instance.bboxes[pred_instance.labels == 0]
                person_scores = pred_instance.scores[pred_instance.labels == 0]

                current_bbox = None
                if len(person_bboxes) > 0:
                    if tracked_bbox is None:
                        best_bbox_idx = np.argmax(person_scores)
                        current_bbox = person_bboxes[best_bbox_idx]
                        tracked_bbox = current_bbox
                    else:
                        ious = [iou(tracked_bbox, box) for box in person_bboxes]
                        best_bbox_idx = np.argmax(ious)
                        if ious[best_bbox_idx] > 0.3:
                            current_bbox = person_bboxes[best_bbox_idx]
                            tracked_bbox = current_bbox
                        else:
                            tracked_bbox = None 
                
                if current_bbox is not None:
                    bboxes_for_pose = np.array([current_bbox])
                    pose_results = inference_topdown(pose_estimator, frame, bboxes_for_pose)
                    data_samples = merge_data_samples(pose_results)

                    predictions = data_samples.get("pred_instances", None)
                    current_frame_predictions = clean_numpy(split_instances(predictions))
                    last_processed_predictions = current_frame_predictions # 更新最後成功處理的預測
                else:
                    current_frame_predictions = []
                    last_processed_predictions = [] # 如果沒有檢測到人，清空之前的預測
            else:
                # 跳過幀，使用上一幀的結果（或者沒有結果）
                current_frame_predictions = last_processed_predictions
            
            if frame_idx < total_frames and total_frames > 0:
                frame_results[frame_idx] = {
                    "frame_idx": frame_idx,
                    "predictions": current_frame_predictions,
                }
            else:
                frame_results.append({
                    "frame_idx": frame_idx,
                    "predictions": current_frame_predictions,
                })
            
            frame_idx += 1
    finally:
        cap.release()
    
    frame_results = [res for res in frame_results if res is not None]

    return JSONResponse(content={"frames": frame_results})