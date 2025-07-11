# 使用 NVIDIA CUDA 11.8、cuDNN 8 的 Ubuntu 22.04 運行時基礎映像。
# 這提供了 GPU 加速所需的底層庫。
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 設定 DEBIAN_FRONTEND 環境變數為 noninteractive，避免在 apt-get 安裝過程中出現互動式提示。
ENV DEBIAN_FRONTEND=noninteractive

# 設定工作目錄，所有後續操作都將在這個目錄下進行，有利於組織檔案。
WORKDIR /workspace

# 更新 apt 軟體包列表並安裝必要的系統依賴。
# 使用 --no-install-recommends 減少映像檔大小。
# python3.10, python3.10-venv, python3-pip, git: Python 開發環境和版本控制工具。
# libgl1, libglib2.0-0, libsm6, libxext6, libxrender1: OpenCV 運行時所需的圖形庫依賴。
# ffmpeg: 處理影片文件所需的工具。
# curl: 用於下載文件。
# 最後，清理 apt 緩存以減少 Docker 映像大小。
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

# 將 /usr/bin/python 指向 /usr/bin/python3.10，確保系統預設使用 Python 3.10。
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# 在 /opt/venv 路徑下創建一個 Python 虛擬環境。
RUN python -m venv /opt/venv
# 將虛擬環境的 bin 目錄添加到 PATH 環境變數中，確保所有 pip 安裝的包都在虛擬環境中。
ENV PATH="/opt/venv/bin:$PATH"

# 升級 pip、setuptools 和 wheel，確保 Python 包管理工具是最新的。
RUN pip install --upgrade pip setuptools wheel

# 安裝 PyTorch、TorchVision 和 TorchAudio，並指定與 CUDA 11.8 兼容的版本。
# --index-url 指向 PyTorch 官方為 CUDA 11.8 編譯的 wheel 文件下載地址。
RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
# 安裝其他必要的 Python 庫：
# "numpy<2.0.0": 限制 NumPy 版本，避免潛在的兼容性問題。
# openmim: OpenMMLab 專案的管理器，用於安裝 MMLab 庫和下載模型。
# fastapi, uvicorn: 用於構建和運行 RESTful API 服務。
# python-multipart: FastAPI 處理文件上傳所需的依賴。
RUN pip install "numpy<2.0.0" openmim fastapi uvicorn python-multipart

# 使用 mim 工具安裝指定版本的 mmcv 和 mmdet。
# mmcv 是 OpenMMLab 的基礎庫，mmdet 是 MMDetection，MMPose 依賴它們。
RUN mim install mmcv==2.0.1
RUN mim install mmdet==3.0.0

# 從 GitHub 克隆 MMPose 儲存庫到容器內的 /workspace/mmpose 目錄。
RUN git clone https://github.com/open-mmlab/mmpose.git /workspace/mmpose
# 將工作目錄設置為 MMPose 專案的根目錄。
WORKDIR /workspace/mmpose
# 安裝 MMPose 專案自身的 Python 依賴，這些依賴列在 requirements.txt 中。
RUN pip install -r requirements.txt
# 以可編輯模式 (-e .) 安裝 MMPose，這樣可以從源代碼直接導入和使用 MMPose 模組。
RUN pip install -v -e .
RUN pip install "numpy<2.0.0"

# mmpose安裝完畢切換回主工作目錄。
WORKDIR /workspace

RUN git clone https://github.com/open-mmlab/mmdetection.git /workspace/mmdetection
WORKDIR /workspace/mmdetection
RUN pip install -r requirements.txt
RUN pip install -v -e .
RUN pip install "numpy<2.0.0"

# mmdetection安裝完畢切換回主工作目錄。
WORKDIR /workspace

# 建立模型檔案的儲存目錄。
RUN mkdir -p checkpoints/mmdet checkpoints/mmpose

# 下載 MMDetection 模型設定檔和權重檔。
# 注意：這些 URL 是從 OpenMMLab 官方 Model Zoo 獲取的，請確保它們是最新的。
RUN curl -L https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.0.0/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py -o mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
 && curl -L https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth -o checkpoints/mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

# 下載 MMPose 模型設定檔和權重檔。
RUN curl -L https://raw.githubusercontent.com/open-mmlab/mmpose/v1.0.0/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py -o mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
 && curl -L https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth -o checkpoints/mmpose/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth

# 將本地的 app.py 檔案複製到 /workspace/app.py。
COPY app.py /workspace/app.py

# 宣告容器將在 8080 端口上監聽。這是一個資訊，用於 Docker 端口映射。
EXPOSE 8080

# 定義啟動容器時執行的命令。
# 這裡假設你的 FastAPI 應用程式主檔案是 app.py，且應用程式實例名為 app。
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]