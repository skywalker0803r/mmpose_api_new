#!/bin/bash
# 啟動 Uvicorn web 伺服器並在前台執行
# 這將是容器的主程序
echo "Starting Uvicorn server..."
uvicorn app:app --host 0.0.0.0 --port 8080