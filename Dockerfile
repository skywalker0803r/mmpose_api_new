FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

RUN pip install \
    "numpy<2.0.0" \
    openmim \
    fastapi \
    uvicorn \
    python-multipart \
    json-tricks \
    opencv-python-headless \
    baseballcv

RUN mim install mmcv==2.0.1

RUN git clone https://github.com/open-mmlab/mmpose.git /workspace/mmpose
WORKDIR /workspace/mmpose
RUN pip install -r requirements.txt
RUN pip install -v -e .

WORKDIR /workspace

RUN mkdir -p checkpoints/mmpose

RUN curl -L https://raw.githubusercontent.com/open-mmlab/mmpose/dev-1.x/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py -o mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py \
 && curl -L https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth -o checkpoints/mmpose/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth

COPY app.py /workspace/app.py
COPY run_inference_baseballcv.py /workspace/run_inference_baseballcv.py

# Set default values for environment variables
ENV POSE_CONFIG='mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py'
ENV POSE_CHECKPOINT='checkpoints/mmpose/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth'
ENV DEVICE='cuda:0'

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]