# MMPose Baseball Pitcher Pose API

This project provides a FastAPI-based web service to analyze the pose of a baseball pitcher from a given video. It uses `baseballcv` to detect the pitcher and `mmpose` (specifically RTMPose) to perform 2D keypoint estimation.

The entire application is containerized using Docker for easy deployment and scaling.

## Prerequisites

- [Docker](https://www.docker.com/get-started)
- NVIDIA GPU with appropriate drivers
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## How to Build the Docker Image

Navigate to the project's root directory and run the following command:

```bash
docker build -t mmpose_api .
```

## How to Run the Docker Container

To run the container with GPU support, execute the command below. This will start the service and map it to port 8080 on your local machine.

```bash
docker run --rm -p 8080:8080 -d --gpus all mmpose_api
```
> The `--rm` flag automatically removes the container when it exits.
> The `-d` flag runs the container in detached mode.

## API Usage

The API exposes a single endpoint `/pose_video` for pose estimation from a video file.

### Endpoint: `POST /pose_video`

Upload a video file to get the pose keypoints for the detected pitcher in each frame.

**Example using `curl`:**

```bash
curl -X POST -F "file=@/path/to/your/video.mp4" http://localhost:8080/pose_video
```

**Successful Response:**

A successful request will return a JSON object containing the pose data for each frame.

```json
{
  "frames": [
    {
      "frame_idx": 0,
      "predictions": {
        "keypoints": [
          [x1, y1, score1],
          [x2, y2, score2],
          ...
        ],
        "bbox": [x1, y1, x2, y2],
        "bbox_score": 0.99
      }
    },
    ...
  ]
}
```

**Error Response:**

If an error occurs during processing (e.g., invalid video file), the API will return a JSON error message with a `500` status code.

```json
{
  "error": "An error occurred: ..."
}
```