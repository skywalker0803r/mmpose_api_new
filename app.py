from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os
from run_inference_baseballcv import run_inference_with_baseballcv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Read settings from environment variables with defaults
POSE_CONFIG = os.environ.get('POSE_CONFIG', 'mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py')
POSE_CHECKPOINT = os.environ.get('POSE_CHECKPOINT', 'checkpoints/mmpose/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth')
DEVICE = os.environ.get('DEVICE', 'cuda:0')

@app.post("/pose_video")
async def pose_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_path = tmp.name
        try:
            content = await file.read()
            tmp.write(content)
        except Exception as e:
            logger.error(f"Failed to read or write uploaded file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process uploaded file: {e}")
    
    background_tasks.add_task(os.remove, video_path)

    try:
        logger.info(f"Starting inference for video: {video_path}")
        result = run_inference_with_baseballcv(
            video_path,
            pose_config=POSE_CONFIG,
            pose_checkpoint=POSE_CHECKPOINT,
            device=DEVICE
        )
        logger.info(f"Inference completed for video: {video_path}")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"An error occurred during inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during inference: {e}")
