#!/bin/bash

# 啟動 Redis 伺服器並在背景執行
redis-server --daemonize yes
echo "Redis server started."

# 等待 Redis 伺服器啟動
sleep 2

# 啟動 Celery worker 並在背景執行
# -A 指向 Celery 應用實例
# --loglevel=info 設定日誌級別
# --concurrency=1 確保每個 worker 只處理一個任務，避免 GPU 資源衝突
celery -A celery_worker.celery_app worker --loglevel=info --concurrency=1 &
echo "Celery worker started."

# 啟動 Uvicorn web 伺服器並在前台執行
# 這將是容器的主程序
echo "Starting Uvicorn server..."
uvicorn app:app --host 0.0.0.0 --port 8080
