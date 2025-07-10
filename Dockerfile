# 使用 NVIDIA CUDA 11.8、cuDNN 8 的 Ubuntu 22.04 運行時基礎映像。
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 設定 DEBIAN_FRONTEND 環境變數為 noninteractive，避免在 apt-get 安裝過程中出現互動式提示。
ENV DEBIAN_FRONTEND=noninteractive

# 設定工作目錄。
WORKDIR /workspace

# 更新 apt 軟體包列表並安裝必要的系統依賴。
# 新增 redis-server 用於任務佇列。
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg curl \
    redis-server \
 && rm -rf /var/lib/apt/lists/*

# 將 /usr/bin/python 指向 /usr/bin/python3.10。
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# 創建並啟用 Python 虛擬環境。
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 升級 pip。
RUN pip install --upgrade pip setuptools wheel

# 安裝 PyTorch。
RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# 安裝其他 Python 庫。
# 新增 celery 和 redis 用於非同步任務處理。
RUN pip install numpy==1.26.4 openmim fastapi uvicorn python-multipart

# 安裝 mmcv 和 mmdet。
RUN mim install mmcv==2.0.1
RUN mim install mmdet==3.0.0

# 克隆並安裝 mmpose。
RUN git clone https://github.com/open-mmlab/mmpose.git /workspace/mmpose
WORKDIR /workspace/mmpose
RUN pip install -r requirements.txt
RUN pip install -v -e .

# 克隆並安裝 mmdetection。
WORKDIR /workspace
RUN git clone https://github.com/open-mmlab/mmdetection.git /workspace/mmdetection
WORKDIR /workspace/mmdetection
RUN pip install -r requirements.txt
RUN pip install -v -e .

# 切換回主工作目錄
WORKDIR /workspace

# 設定放權重的資料夾
RUN mkdir -p checkpoints/mmdet checkpoints/mmpose

# 下載模型到指定路徑
RUN curl -L https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.1.0/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py -o mmdetection/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py
RUN curl -L https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth -o checkpoints/mmdet/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth
RUN curl -L https://raw.githubusercontent.com/open-mmlab/mmpose/refs/heads/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py -o mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py 
RUN curl -L https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth -o checkpoints/mmpose/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth

 # 複製應用程式檔案到容器中。
COPY app.py /workspace/app.py
COPY start.sh /workspace/start.sh

# 賦予啟動腳本執行權限。
RUN chmod +x /workspace/start.sh

# 宣告容器將在 8080 端口上監聽。
EXPOSE 8080

# 使用啟動腳本來啟動所有服務。
CMD ["/workspace/start.sh"]
