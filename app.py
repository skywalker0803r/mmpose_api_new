import os
import tempfile
import cv2
import numpy as np
import sys
import uuid # 用於生成唯一的任務 ID

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# 導入 mmpose 和 mmdet 的核心函式
from mmpose.apis import init_model as init_pose_estimator, inference_topdown
from mmdet.apis import init_detector, inference_detector
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

# 確保使用 'spawn' 啟動方法進行多程序處理，這對於在多程序環境中使用 CUDA 至關重要。
# 必須在任何 CUDA 操作或子程序創建之前呼叫此函數。
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # 捕獲 RuntimeError 以防 set_start_method 已被呼叫 (例如在其他模組中)
    pass

print("--- FastAPI Web App 正在啟動 ---")

# --- 1. 建立 FastAPI 應用 ---
app = FastAPI(title="MMPose Synchronous API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. 模型初始化 (使用 RTMPose 和 RTMDet) ---
# 模型只在應用程式啟動時載入一次，而不是每次請求都載入
# 使用更快的 RTMDet 進行人體偵測
det_config = 'mmdetection/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py'
det_checkpoint = 'checkpoints/mmdet/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'

# 使用更快的 RTMPose 進行姿勢估計
pose_config = 'mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py'
pose_checkpoint = 'checkpoints/mmpose/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth'

detector = None
pose_estimator = None

@app.on_event("startup")
async def load_models():
    """
    在 FastAPI 應用程式啟動時載入模型。
    """
    global detector, pose_estimator
    try:
        print("[初始化] 正在載入模型...")
        # 初始化偵測器模型，並指定使用 CUDA 設備
        detector = init_detector(det_config, det_checkpoint, device='cuda:0')
        detector.cfg = adapt_mmdet_pipeline(detector.cfg)
        # 初始化姿勢估計器模型，並指定使用 CUDA 設備
        pose_estimator = init_pose_estimator(pose_config, pose_checkpoint, device='cuda:0')
        print("✅ 所有模型初始化成功！")
    except Exception as e:
        print(f"❌ 初始化過程中發生嚴重錯誤: {e}")
        # 在生產環境中，這裡可以考慮更優雅的錯誤處理，例如記錄日誌並退出應用程式
        sys.exit(1) # 初始化失敗時退出程序

# --- 輔助函式 ---
def clean_numpy(obj):
    """
    遞歸地將 NumPy 數據類型轉換為標準 Python 數據類型，以便 JSON 序列化。
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: clean_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_numpy(i) for i in obj]
    return obj

def iou(box1, box2):
    """
    計算兩個邊界框之間的 IoU (Intersection over Union)。
    邊界框格式: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box1[0], box1[1], box1[2], box1[3]
    x3, y3, x4, y4 = box2[0], box2[1], box2[2], box2[3]

    # 計算交集區域的座標
    inter_x1, inter_y1 = max(x1, x3), max(y1, y3)
    inter_x2, inter_y2 = min(x2, x4), min(y2, y4)

    # 計算交集區域的寬度和高度
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # 計算兩個邊界框的面積
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)

    # 計算聯集區域
    union_area = area1 + area2 - inter_area

    # 避免除以零
    return inter_area / union_area if union_area > 0 else 0

# --- 3. API 路由 (Endpoints) ---

@app.get("/")
def read_root():
    """根目錄，提供歡迎訊息。"""
    return {"message": "歡迎使用 MMPose API！"}

@app.post("/pose_video")
async def process_video_sync(file: UploadFile = File(...)):
    """
    接收影片檔案，並同步處理姿勢估計。
    處理完成後直接返回結果。
    """
    if detector is None or pose_estimator is None:
        raise HTTPException(status_code=503, detail="模型尚未載入完成，請稍後再試。")

    temp_video_path = None
    try:
        # 將上傳的檔案存到一個臨時檔案中
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_path = temp_video_file.name
            content = await file.read()
            temp_video_file.write(content)

        print(f"開始處理影片: {temp_video_path}")

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="無法開啟影片檔案。")

        frame_idx = 0
        frame_results = []
        tracked_bbox = None # 用於追蹤主要人物的邊界框
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                # 如果無法讀取更多幀，則退出循環
                break
            
            print(f"正在處理第 {frame_idx}/{total_frames} 幀...")

            # 執行人體偵測
            det_result = inference_detector(detector, frame)
            pred_instance = det_result.pred_instances.cpu().numpy()
            
            # 過濾出人物的邊界框和分數 (假設標籤 0 是人)
            person_bboxes = pred_instance.bboxes[pred_instance.labels == 0]
            person_scores = pred_instance.scores[pred_instance.labels == 0]

            current_bbox = None
            if len(person_bboxes) > 0:
                if tracked_bbox is None:
                    # 如果是第一幀或之前沒有追蹤到人物，選擇分數最高的人物
                    best_bbox_idx = np.argmax(person_scores)
                    current_bbox = person_bboxes[best_bbox_idx]
                else:
                    # 如果之前有追蹤到人物，計算與所有檢測到人物的 IoU
                    ious = [iou(tracked_bbox, box) for box in person_bboxes]
                    best_bbox_idx = np.argmax(ious)
                    # 如果最佳 IoU 大於閾值，則更新追蹤的邊界框
                    if ious[best_bbox_idx] > 0.3:
                        current_bbox = person_bboxes[best_bbox_idx]
            
            # 更新追蹤的邊界框：如果當前幀檢測到人物，則更新為當前人物的邊界框；否則重置為 None
            if current_bbox is not None:
                tracked_bbox = current_bbox
            else:
                tracked_bbox = None # 如果沒有檢測到合適的人物，則重置追蹤

            if current_bbox is None:
                # 如果當前幀沒有檢測到或追蹤到人物，則記錄空結果並跳過姿勢估計
                frame_results.append({"frame_idx": frame_idx, "predictions": []})
                frame_idx += 1
                continue # 跳到下一幀

            # 對追蹤到的單個人物執行姿勢估計
            bboxes_for_pose = [current_bbox]
            pose_results = inference_topdown(pose_estimator, frame, bboxes_for_pose)
            data_samples = merge_data_samples(pose_results)
            
            # 提取姿勢估計結果並進行清理
            predictions = data_samples.get("pred_instances", None)
            split_preds = split_instances(predictions)
            cleaned_preds = clean_numpy(split_preds)

            # 將當前幀的結果添加到列表中
            frame_results.append({
                "frame_idx": frame_idx,
                "predictions": cleaned_preds,
            })
            
            frame_idx += 1

    except Exception as e:
        print(f"處理過程中發生錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"影片處理失敗: {e}")
    finally:
        # 無論成功或失敗，都釋放影片資源並刪除臨時檔案
        if cap.isOpened():
            cap.release()
        if temp_video_path and os.path.exists(temp_video_path):
            print(f"刪除臨時檔案: {temp_video_path}")
            os.remove(temp_video_path)

    print("影片處理完成。")
    return JSONResponse(status_code=200, content={"frames": frame_results})

print("🚀 FastAPI Web App 已準備就緒！請訪問 http://localhost:8000/docs")
