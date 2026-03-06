from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from PIL import Image
import io, os, uuid, tempfile, subprocess
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms
import librosa, numpy as np
import cv2
from pathlib import Path
import json
from fastapi.middleware.cors import CORSMiddleware


# ------------------ shared config ------------------
IMG_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUDIO_DEVICE = IMG_DEVICE
AUDIO_SR = 16000

# ==== AUDIO CONFIG (copy these under AUDIO_SR) ====
CFG = {
    "AUDIO_CLASS_ORDER": ["REAL", "FAKE"],   # <-- set to your training order; flip if needed
    "MEL_PARAMS": {                          # <-- EXACT mel/FFT params from training
        "n_mels": 64,
        "n_fft": 2048,
        "hop_length": 512,
        "win_length": 2048,
        "fmin": 0,
        "fmax": None,
        "center": True,
    },
    "AUDIO_FLIP_UD": False,                  # do NOT flip; training saved low-freqs at top
    "AUDIO_MINMAX_PER_IMAGE": True,    
    
    "DB_REF": "max",                         # "max" if ref=np.max; "one" if ref=1.0
    "THRESH": 0.7,                           # decision threshold for FAKE
    "AGG": "median",                         # "median" or "topk_mean"
    "TOPK": 3,                               # used when AGG == "topk_mean"
}

# ------------------ image model ------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

img_model = SimpleCNN().to(IMG_DEVICE)
img_model.load_state_dict(torch.load("best_model.pth", map_location=IMG_DEVICE))
img_model.eval()

img_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ------------------ face detector (same as preprocessing) ------------------
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def crop_face_from_rgb(rgb):
    """
    rgb: HxWx3 np.uint8 in RGB
    returns: PIL.Image (face crop) or None if not found
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None

    # pick largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # add a small margin like you likely had from cascade boxes during preprocessing
    m = int(0.10 * max(w, h))
    x0 = max(0, x - m)
    y0 = max(0, y - m)
    x1 = min(rgb.shape[1], x + w + m)
    y1 = min(rgb.shape[0], y + h + m)
    face = rgb[y0:y1, x0:x1]

    return Image.fromarray(face)

THRESH_IMG = 0.5

# ------------------ audio model ------------------
class SimpleAudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool  = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)

        self.fc1   = nn.Linear(32 * 32 * 32, 128)
        self.bn3   = nn.BatchNorm1d(128)
        self.fc2   = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.bn3(self.fc1(x)))
        return self.fc2(x)

AUDIO_WEIGHTS = "best_audio_model.pth"  # make sure this file exists
audio_model = SimpleAudioCNN().to(AUDIO_DEVICE)
audio_model.load_state_dict(torch.load(AUDIO_WEIGHTS, map_location=AUDIO_DEVICE), strict=True)
audio_model.eval()


AUDIO_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # no ImageNet normalization unless you trained with it
])

def _spec_from_wave(y: np.ndarray) -> Image.Image:
    """Replicate training spectrogram -> PIL image (RGB) with per-image min-max."""
    mp = CFG["MEL_PARAMS"]
    S = librosa.feature.melspectrogram(y=y, sr=AUDIO_SR,
                                       **{k: v for k, v in mp.items() if v is not None})
    # same ref as training convert_audio_to_spectrogram()
    S_db = librosa.power_to_db(S, ref=np.max)

    # training used origin='lower' -> low freq at bottom; so NO flip
    if CFG["AUDIO_FLIP_UD"]:
        S_db = np.flipud(S_db)

    # per-image min-max (what imshow(gray) effectively does for PNGs)
    S_min, S_max = S_db.min(), S_db.max()
    if S_max - S_min < 1e-8:
        S_img = np.zeros_like(S_db, dtype=np.uint8)
    else:
        S_img = ((S_db - S_min) / (S_max - S_min) * 255.0).astype(np.uint8)

    # grayscale -> RGB (your model expects 3 channels)
    return Image.fromarray(S_img, mode="L").convert("RGB")

def wav_to_single_spec_tensor(wav_path: str) -> tuple[torch.Tensor, Image.Image]:
    """Load WAV -> single spectrogram tensor [1, 3, 128, 128] and return the PIL image for optional debug."""
    y, _ = librosa.load(wav_path, sr=AUDIO_SR, mono=True)
    if y.size == 0:
        # avoid empty input
        y = np.zeros(int(1.0 * AUDIO_SR), dtype=np.float32)

    img = _spec_from_wave(y)                 # RGB PIL image
    x = AUDIO_TRANSFORM(img).unsqueeze(0)    # [1, 3, 128, 128]
    return x, img

def _save_debug_spec_image(img: Image.Image, out_dir="/tmp/audio_debug", name="inference_spec.png"):
    os.makedirs(out_dir, exist_ok=True)
    img.save(os.path.join(out_dir, name))

def _make_mel_image(y: np.ndarray) -> Image.Image:
    mp = CFG["MEL_PARAMS"]
    S = librosa.feature.melspectrogram(y=y, sr=AUDIO_SR, **{k:v for k,v in mp.items() if v is not None})
    S_db = librosa.power_to_db(S, ref=np.max)  # same as training

    # Do not flip (origin='lower' in training already handled orientation)
    # Do per-image min-max scaling like matplotlib would when saving grayscale PNGs
    S_min, S_max = S_db.min(), S_db.max()
    if S_max - S_min < 1e-8:
        S_img = np.zeros_like(S_db, dtype=np.uint8)
    else:
        S_img = ((S_db - S_min) / (S_max - S_min) * 255.0).astype(np.uint8)

    # grayscale -> RGB; we'll resize + normalize in AUDIO_TRANSFORM
    return Image.fromarray(S_img, mode="L").convert("RGB")

def wav_to_spec_tensor_batch(wav_path: str) -> torch.Tensor:
    win = int(CFG["AUDIO_WIN_SEC"] * AUDIO_SR)
    hop = int(CFG["AUDIO_HOP_SEC"] * AUDIO_SR)

    y, _ = librosa.load(wav_path, sr=AUDIO_SR, mono=True)
    if y.size == 0:
        y = np.zeros(win, dtype=np.float32)

    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
        chunks = [y]
    else:
        chunks = []
        for start in range(0, max(1, len(y) - win + 1), hop):
            seg = y[start:start+win]
            if len(seg) < win:
                seg = np.pad(seg, (0, win - len(seg)))
            chunks.append(seg)

    imgs = [_make_mel_image(seg) for seg in chunks]
    tensors = [AUDIO_TRANSFORM(img) for img in imgs]   # 224 + ImageNet norm
    return torch.stack(tensors, dim=0)                 # [N, 3, 224, 224]

def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{err}")

def _ffprobe_has_audio(input_path: str) -> bool:
    probe = [
        "ffprobe", "-v", "error",
        "-show_streams", "-select_streams", "a",
        "-of", "json", input_path
    ]
    p = subprocess.run(probe, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr.strip() or 'unknown error'}")
    info = json.loads(p.stdout or "{}")
    return bool(info.get("streams"))

def _extract_wav_16k_mono(input_path: str) -> str:
    in_p = Path(input_path)
    # If input already ends with .wav, avoid same input/output path
    if in_p.suffix.lower() == ".wav":
        out_p = in_p.with_name(in_p.stem + "_16kmono.wav")
    else:
        out_p = in_p.with_suffix(".wav")

    cmd = [
        "ffmpeg", "-nostdin", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", str(in_p),
        "-vn", "-ar", str(AUDIO_SR), "-ac", "1",
        "-f", "wav", str(out_p),
    ]
    _run(cmd)
    return str(out_p)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# ------------------ image endpoints ------------------
@app.post("/predict_frame")
async def predict_frame(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    x = img_transform(img).unsqueeze(0).to(IMG_DEVICE)
    with torch.inference_mode():
        logits = img_model(x)
        prob_fake = torch.softmax(logits, dim=1)[0, 1].item()
    pred = int(prob_fake >= THRESH_IMG)
    return {"prob_fake": prob_fake, "pred": pred, "label_map": {"0":"REAL","1":"FAKE"}}

THRESH_IMG_VIDEO = 0.5  # you can tune this to 0.55–0.65 after checking a few videos

@app.post("/predict_video")
async def predict_video(
    file: UploadFile = File(...),
    fps_interval: float = 1.0,
    max_frames: int = 64,
    min_face_frames: int = 8,      # require at least N frames with a found face
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video")

        frame_rate = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(frame_rate * fps_interval))

        face_imgs = []
        idx = 0
        sampled = 0

        while sampled < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx % step == 0:
                # BGR -> RGB
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # crop face like training
                pil_face = crop_face_from_rgb(frame)

                # Fallback: if no face, skip the frame (better than feeding background)
                if pil_face is not None:
                    face_imgs.append(pil_face)
                    sampled += 1
            idx += 1

        cap.release()

        if len(face_imgs) < max(1, min_face_frames):
            # As a last resort you can center-crop a few frames, but skipping is safer.
            return {
                "error": "not_enough_faces",
                "message": f"Found faces on {len(face_imgs)} frames; need at least {min_face_frames} to score reliably."
            }

        # batch predict
        batch = torch.stack([img_transform(img) for img in face_imgs], dim=0).to(IMG_DEVICE)

        with torch.inference_mode():
            logits = img_model(batch)                         # [N, 2]
            probs_fake = torch.softmax(logits, dim=1)[:, 1]   # P(fake) per frame

        # robust aggregation (median + top-k mean)
        pf = probs_fake.detach().cpu().numpy()
        prob_fake_mean   = float(np.mean(pf))
        prob_fake_median = float(np.median(pf))
        topk = min(10, len(pf))
        prob_fake_topk_mean = float(np.mean(np.sort(pf)[-topk:]))

        # majority vote across frames after thresholding
        frame_preds = (pf >= THRESH_IMG_VIDEO).astype(np.int32)
        frac_frames_fake = float(frame_preds.mean())

        # final decision – majority vote usually works better than plain mean
        pred = int(frac_frames_fake >= 0.5)

        # some debug stats back
        return {
            "pred": pred,
            "threshold_used": THRESH_IMG_VIDEO,
            "frames_with_face": len(face_imgs),
            "prob_fake_mean": prob_fake_mean,
            "prob_fake_median": prob_fake_median,
            "prob_fake_topk_mean": prob_fake_topk_mean,
            "frac_frames_pred_fake": frac_frames_fake,
            "label_map": {"0":"REAL","1":"FAKE"},
            "sample_frame_probs_fake": [float(x) for x in pf[:10]],
        }

    finally:
        try: os.remove(tmp_path)
        except: pass

# ------------------ audio endpoints ------------------
THRESH_AUDIO = 0.7

@app.post("/predict_audio")
async def predict_audio(file: UploadFile = File(...), debug: bool = False):
    # Save upload to a CLOSED temp file with the right suffix
    suffix = Path(file.filename or "audio").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name  # closed when we exit the 'with'

    wav_path = None
    try:
        if suffix.lower() != ".wav":
            # Re-encode to 16k mono WAV
            out_p = Path(tmp_path).with_suffix(".wav")
            _run([
                "ffmpeg", "-nostdin", "-y",
                "-hide_banner", "-loglevel", "error",
                "-i", tmp_path,
                "-ar", str(AUDIO_SR), "-ac", "1",
                "-f", "wav", str(out_p),
            ])
            wav_path = str(out_p)
        else:
            # Ensure consistent format even if .wav came in
            wav_path = _extract_wav_16k_mono(tmp_path)

        # Build a single spectrogram tensor (NO CHUNKING)
        x, img_for_debug = wav_to_single_spec_tensor(wav_path)
        x = x.to(AUDIO_DEVICE)  # [1, 3, 128, 128]

        # Inference
        with torch.inference_mode():
            logits = audio_model(x)     # [1, 2]
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        # class indices based on training order
        idx_real = CFG["AUDIO_CLASS_ORDER"].index("REAL")  # 0
        idx_fake = CFG["AUDIO_CLASS_ORDER"].index("FAKE")  # 1

        p_fake = float(probs[idx_fake])
        p_real = float(probs[idx_real])
        pred   = 1 if p_fake >= CFG["THRESH"] else 0  # 1=FAKE, 0=REAL

        # optional debug: save the exact spectrogram used
        if debug:
            _save_debug_spec_image(img_for_debug, name="inference_spec.png")

        return {
            "pred": pred,
            "prob_fake": p_fake,
            "prob_real": p_real,
            "threshold_used": CFG["THRESH"],
            "label_map": {"0": "REAL", "1": "FAKE"},
            "chunks": 1,   # always 1 now
            "config_used": {
                "AUDIO_CLASS_ORDER": CFG["AUDIO_CLASS_ORDER"],
                "MEL_PARAMS": CFG["MEL_PARAMS"],
                "AUDIO_FLIP_UD": CFG["AUDIO_FLIP_UD"],
                "AUDIO_MINMAX_PER_IMAGE": CFG["AUDIO_MINMAX_PER_IMAGE"],
                "DB_REF": CFG["DB_REF"],
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in (tmp_path, wav_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except:
                pass

# ------------------ unified prediction endpoint ------------------
from fastapi import UploadFile, File

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = file.filename.lower()
    ext = Path(filename).suffix

    if ext in [".jpg", ".jpeg", ".png"]:
        # Image
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        x = img_transform(img).unsqueeze(0).to(IMG_DEVICE)
        with torch.inference_mode():
            logits = img_model(x)
            prob_fake = torch.softmax(logits, dim=1)[0, 1].item()
        pred = int(prob_fake >= THRESH_IMG)
        return {
            "media_type": "image",
            "filename": filename,
            "prediction": {
                "pred": pred,
                "prob_fake": prob_fake,
                "prob_real": 1 - prob_fake,
                "label_map": {"0": "REAL", "1": "FAKE"},
            }
        }

    elif ext in [".mp4", ".mov"]:
        res = await predict_video(file)
        return {
            "media_type": "video",
            "filename": filename,
            "prediction": res
        }

    elif ext in [".mp3", ".wav"]:
        res = await predict_audio(file)
        return {
            "media_type": "audio",
            "filename": filename,
            "prediction": res
        }


    else:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")