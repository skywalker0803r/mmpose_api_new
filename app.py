import os
import sys
import tempfile
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from mmpose.apis import init_model as init_pose_estimator, inference_topdown
from mmdet.apis import init_detector, inference_detector
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

from bytetrack.byte_tracker import BYTETracker  # ByteTrack追蹤器

print("--- 啟動 FastAPI + ByteTrack 批次推論 範例 ---")

# 1. 模型初始化
det_config = 'mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
det_checkpoint = 'checkpoints/mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
pose_config = 'mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
pose_checkpoint = 'checkpoints/mmpose/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

try:
    print("[初始化] 載入模型...")
    detector = init_detector(det_config, det_checkpoint, device='cuda:0')
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_estimator = init_pose_estimator(pose_config, pose_checkpoint, device='cuda:0')
    print("✅ 模型載入成功")
except Exception as e:
    print(f"❌ 初始化錯誤: {e}")
    sys.exit()

# 2. 建立 ByteTrack 追蹤器（參數可依需求調整）
tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30)

# 3. 建立 FastAPI 與 CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_numpy(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)): return int(obj)
    if isinstance(obj, dict): return {k: clean_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list): return [clean_numpy(i) for i in obj]
    return obj

@app.post("/pose_video")
async def pose_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    # 暫存影片
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_path = tmp.name
        tmp.write(await file.read())
    background_tasks.add_task(os.remove, video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return JSONResponse(status_code=500, content={"error": "無法開啟影片"})

    frame_idx = 0
    frame_results = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            print(f"處理第 {frame_idx} 幀")

            # 物件偵測（偵測所有人）
            det_result = inference_detector(detector, frame)
            pred_instance = det_result.pred_instances.cpu().numpy()
            # 選擇 label=0 (person)
            person_bboxes = pred_instance.bboxes[pred_instance.labels == 0]
            person_scores = pred_instance.scores[pred_instance.labels == 0]

            # 整理成 ByteTrack 輸入格式 [x1,y1,x2,y2,score]
            dets_for_tracking = []
            for bbox, score in zip(person_bboxes, person_scores):
                dets_for_tracking.append([bbox[0], bbox[1], bbox[2], bbox[3], score])
            dets_for_tracking = np.array(dets_for_tracking)

            # ByteTrack 追蹤結果 (tracks: 有 id, bbox)
            online_targets = tracker.update(dets_for_tracking, [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]])
            
            # 把追蹤結果 bbox 裁切，準備批次推論姿態
            tracked_bboxes = []
            track_ids = []
            for t in online_targets:
                tlwh = t.tlwh  # 左上寬高
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                # 限制邊界
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
                tracked_bboxes.append([x1, y1, x2, y2])
                track_ids.append(t.track_id)

            if len(tracked_bboxes) == 0:
                frame_results.append({"frame_idx": frame_idx, "predictions": []})
                frame_idx += 1
                continue

            # 批次姿態推論
            pose_results = inference_topdown(pose_estimator, frame, tracked_bboxes)
            data_samples = merge_data_samples(pose_results)
            predictions = data_samples.get("pred_instances", None)
            split_preds = split_instances(predictions)
            cleaned_preds = clean_numpy(split_preds)

            # 包含追蹤ID一起回傳 (將cleaned_preds與track_ids對應)
            # 假設split_preds結構為 list，每一項為一個目標的預測
            # 這邊做簡單對應，實際可依需要調整
            for i, pred in enumerate(cleaned_preds):
                pred["track_id"] = track_ids[i]

            frame_results.append({
                "frame_idx": frame_idx,
                "predictions": cleaned_preds,
            })

            frame_idx += 1

    finally:
        cap.release()

    return JSONResponse(content={"frames": frame_results})
