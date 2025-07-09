
import cv2
import numpy as np
import os
import sys
import tempfile
from celery import Celery
from celery.utils.log import get_task_logger

# 導入 mmpose 和 mmdet 的核心函式
from mmpose.apis import init_model as init_pose_estimator, inference_topdown
from mmdet.apis import init_detector, inference_detector
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

print("--- Celery Worker 正在啟動 ---")

# --- 1. Celery 設定 ---
# 使用 Redis 作為訊息代理 (Broker) 和結果後端 (Backend)
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)
celery_app.conf.update(
    task_track_started=True
)
logger = get_task_logger(__name__)

# --- 2. 模型初始化 (使用 RTMPose 和 RTMDet) ---
# 模型只在 Worker 啟動時載入一次，而不是每次請求都載入
# 使用更快的 RTMDet 進行人體偵測
det_config = 'mmdetection/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py'
det_checkpoint = 'checkpoints/mmdet/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'

# 使用更快的 RTMPose 進行姿勢估計
pose_config = 'mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py'
pose_checkpoint = 'checkpoints/mmpose/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth'

try:
    logger.info("[初始化] 正在載入模型...")
    detector = init_detector(det_config, det_checkpoint, device='cuda:0')
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_estimator = init_pose_estimator(pose_config, pose_checkpoint, device='cuda:0')
    logger.info("✅ 所有模型初始化成功！")
except Exception as e:
    logger.error(f"❌ 初始化過程中發生嚴重錯誤: {e}")
    sys.exit()

# --- 輔助函式 (與原 app.py 相同) ---
def clean_numpy(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)): return int(obj)
    if isinstance(obj, dict): return {k: clean_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list): return [clean_numpy(i) for i in obj]
    return obj

def iou(box1, box2):
    x1, y1, x2, y2 = box1[0], box1[1], box1[2], box1[3]
    x3, y3, x4, y4 = box2[0], box2[1], box2[2], box2[3]
    inter_x1, inter_y1 = max(x1, x3), max(y1, y3)
    inter_x2, inter_y2 = min(x2, x4), min(y2, y4)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1, area2 = (x2 - x1) * (y2 - y1), (x4 - y3) * (y4 - y3)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

# --- 3. 定義 Celery 任務 ---
@celery_app.task(bind=True)
def process_video_task(self, video_path: str):
    """
    這是一個 Celery 背景任務，負責處理影片的姿勢估計。
    """
    logger.info(f"[{self.request.id}] 開始處理影片: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"[{self.request.id}] 無法開啟影片檔案: {video_path}")
        # 刪除臨時檔案
        if os.path.exists(video_path):
            os.remove(video_path)
        raise IOError("無法開啟影片檔案")

    frame_idx = 0
    frame_results = []
    tracked_bbox = None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # 更新任務狀態，提供進度
            self.update_state(state='PROGRESS', meta={'current': frame_idx, 'total': total_frames})
            logger.info(f"[{self.request.id}] 正在處理第 {frame_idx}/{total_frames} 幀...")

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
            
            if current_bbox is None:
                frame_results.append({"frame_idx": frame_idx, "predictions": []})
                frame_idx += 1
                continue

            bboxes_for_pose = [current_bbox]
            pose_results = inference_topdown(pose_estimator, frame, bboxes_for_pose)
            data_samples = merge_data_samples(pose_results)
            
            predictions = data_samples.get("pred_instances", None)
            split_preds = split_instances(predictions)
            cleaned_preds = clean_numpy(split_preds)

            frame_results.append({
                "frame_idx": frame_idx,
                "predictions": cleaned_preds,
            })
            
            frame_idx += 1
    except Exception as e:
        logger.error(f"[{self.request.id}] 處理過程中發生錯誤: {e}")
        raise e # 重新拋出異常，讓 Celery 知道任務失敗
    finally:
        cap.release()
        # 處理完畢後刪除臨時影片檔案
        if os.path.exists(video_path):
            logger.info(f"[{self.request.id}] 刪除臨時檔案: {video_path}")
            os.remove(video_path)

    logger.info(f"[{self.request.id}] 影片處理完成。")
    return {"frames": frame_results}

print("✅ Celery Worker 已準備就緒！")
