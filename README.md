# Multimodal Deepfake Detection

A practical multimodal deepfake detection project that combines **image**, **video**, and **audio** analysis using CNN-based pipelines and a deployable **FastAPI** backend.

## Why This Project
Deepfake detection is strongest when signals from multiple modalities are analyzed together. This repository provides:

- A reproducible data-preparation pipeline (video -> frames -> face crops -> processed data)
- CNN-based classifiers for visual and audio cues
- Evaluation and explainability notebooks
- API endpoints for production-style inference on uploaded media

## Core Features
- Image deepfake prediction from single frames
- Video deepfake prediction via frame sampling + face detection + temporal aggregation
- Audio deepfake prediction from mel-spectrograms
- Unified endpoint for auto-routing by file type
- FaceForensics++ downloader script for dataset setup

## Repository Highlights
- `forfastapi.py`: FastAPI app with image/video/audio inference endpoints
- `extract_frames.py`: Extracts frames from real/fake video folders
- `split_frames.py`: Splits frames into train/val/test
- `face_detect_crop.py`: Detects and crops faces using OpenCV Haar cascades
- `data_preprocess.py`: Resizes cropped faces to model input size
- `labeling.py`: Builds label CSV from prepared image folders
- `faceforensics_download_v4.py`: FaceForensics++ download utility
- `revealed.ipynb`: Main experimentation/training notebook
- `model_evaluation_and_explainability.ipynb`: Model analysis and explainability workflow
- `video_baseline.ipynb`: Baseline video experiments

## End-to-End Pipeline
1. Download FaceForensics++ data
2. Extract frames from real/fake videos
3. Split frames into train/val/test
4. Detect and crop faces
5. Resize/process images for model input
6. Train and evaluate models (notebooks)
7. Serve inference via FastAPI

## Data Layout (Expected)

```text
data/
  faceforensics/
    original_sequences/youtube/c23/videos
    manipulated_sequences/DeepFakeDetection/c23/videos
  extracted_frames/
    real/
    fake/
  split_frames/
    train/{real,fake}
    val/{real,fake}
    test/{real,fake}
  cropped/
    train/{real,fake}
    val/{real,fake}
  processed/
    train/{real,fake}
    val/{real,fake}
```

## Setup
### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install torch torchvision fastapi uvicorn pillow numpy opencv-python librosa pandas tqdm python-multipart
```

### 3. Install FFmpeg (required for audio/video API paths)

```bash
# macOS (Homebrew)
brew install ffmpeg
```

## Dataset Download (FaceForensics++)

```bash
python faceforensics_download_v4.py data/faceforensics -d original -c c23 -t videos
python faceforensics_download_v4.py data/faceforensics -d DeepFakeDetection -c c23 -t videos
```

You can check all options with:

```bash
python faceforensics_download_v4.py -h
```

## Data Preparation Commands
Run these scripts in order:

```bash
python extract_frames.py
python split_frames.py
python face_detect_crop.py
python data_preprocess.py --mode all
python labeling.py
```

## Model Artifacts
Current repository includes artifacts such as:

- `revealed_deepfake_detector.pth`
- `audio_model.pth`
- `image_simplecnn.onnx`
- `image_simplecnn_ts.pt`

`forfastapi.py` currently loads:

- `best_model.pth` (image)
- `best_audio_model.pth` (audio)

If your checkpoint names differ, either rename files or update the model paths in `forfastapi.py`.

## Running the API

```bash
uvicorn forfastapi:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://localhost:8000/healthz
```

## API Endpoints
- `POST /predict_frame` -> image-only prediction
- `POST /predict_video` -> video prediction using sampled face frames
- `POST /predict_audio` -> audio prediction via mel-spectrogram
- `POST /predict` -> unified endpoint (image/video/audio by extension)

### Example: Unified Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample.mp4"
```

### Example: Audio Prediction

```bash
curl -X POST "http://localhost:8000/predict_audio" \
  -F "file=@sample.wav"
```

## Visual Outputs
Training/evaluation artifacts in this project include:

- `training_curves.png`
- `loss_curve_image_video.png`
- `acc_curve_image_video.png`
- `prediction_distribution.png`
- `cnn_firstconv_filters.png`
- `vid_activations_grid.png`
- `vid_activations_time.png`

## Notebooks
- `revealed.ipynb`: Main training and experimentation flow
- `model_evaluation_and_explainability.ipynb`: Error analysis, interpretability, result inspection
- `video_baseline.ipynb`: Baseline comparisons for video

## Troubleshooting
- If API startup fails with missing model files: verify checkpoint filenames/paths in `forfastapi.py`.
- If audio endpoints fail: ensure `ffmpeg` and `ffprobe` are installed and available in PATH.
- If face detection returns too few faces on videos: tune `fps_interval`, `max_frames`, and `min_face_frames` in `/predict_video`.
- If large files fail to push: keep datasets/checkpoints outside Git or use Git LFS.

## Suggested Next Improvements
- Add `requirements.txt` for exact reproducibility
- Add model training scripts (currently notebook-centric)
- Add unit tests for preprocessing and API contracts
- Add Dockerfile for one-command deployment

## Author
**Shania Khan**

If you use this repository in research or demos, consider citing the project and linking back to the GitHub repo.
