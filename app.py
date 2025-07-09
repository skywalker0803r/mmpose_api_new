import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# 從 celery_worker 導入 Celery app 實例和任務
from celery_worker import celery_app, process_video_task
from celery.result import AsyncResult

print("--- FastAPI Web App 正在啟動 ---")

# --- 1. 建立 FastAPI 應用 ---
app = FastAPI(title="MMPose Asynchronous API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. API 路由 (Endpoints) ---

@app.get("/")
def read_root():
    """根目錄，提供歡迎訊息。"""
    return {"message": "歡迎使用非同步 MMPose API！"}

@app.post("/pose_video", status_code=202)
async def submit_pose_video(file: UploadFile = File(...)):
    """
    接收影片檔案，將其交給背景任務處理，並立即返回任務 ID。
    """
    # 將上傳的檔案存到一個臨時檔案中，Celery Worker 會讀取這個檔案
    # 使用 delete=False 確保在 with 區塊結束後檔案不會被自動刪除
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            video_path = temp_video_file.name
            content = await file.read()
            temp_video_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"無法儲存上傳的檔案: {e}")

    # 呼叫 Celery 任務，將影片路徑作為參數
    # .delay() 會將任務發送到佇列並立即返回
    task = process_video_task.delay(video_path)

    # 返回任務 ID，客戶端可以用它來查詢結果
    return {"task_id": task.id}

@app.get("/results/{task_id}")
def get_task_result(task_id: str):
    """
    根據任務 ID 查詢影片處理的結果或狀態。
    """
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.ready():
        # 任務已完成
        if task_result.successful():
            result = task_result.get()
            return JSONResponse(status_code=200, content=result)
        else:
            # 任務執行失敗
            error_info = str(task_result.info) # 獲取異常信息
            return JSONResponse(status_code=500, content={"status": "FAILURE", "error": error_info})
    else:
        # 任務仍在進行中
        response = {
            "status": task_result.status,
        }
        if task_result.info:
             # 如果任務有回報進度 (在 worker 中使用 update_state)
            response.update(task_result.info)
        return JSONResponse(status_code=202, content=response)

print("🚀 FastAPI Web App 已準備就緒！請訪問 http://localhost:8080/docs")
