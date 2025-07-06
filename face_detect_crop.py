import cv2
import os
from pathlib import Path

def detect_and_crop_face(input_dir, output_dir):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_list = os.listdir(input_dir)
    total = len(image_list)
    processed = 0

    for img_name in image_list:
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)

        # Skip if already processed
        if os.path.exists(output_path):
            processed += 1
            continue

        img = cv2.imread(input_path)
        if img is None:
            print(f"[⚠️] Failed to read {img_name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (224, 224))
            cv2.imwrite(output_path, face_resized)
            break  # Only first detected face

        processed += 1
        if processed % 50 == 0 or processed == total:
            print(f"[✔️] Processed {processed}/{total} images in {input_dir}")

# Run it on all datasets
datasets = [
    ('data/split_frames/train/fake', 'data/cropped/train/fake'),
    ('data/split_frames/train/real', 'data/cropped/train/real'),
    ('data/split_frames/val/fake', 'data/cropped/val/fake'),
    ('data/split_frames/val/real', 'data/cropped/val/real')
]

for input_dir, output_dir in datasets:
    print(f"🔄 Starting: {input_dir}")
    detect_and_crop_face(input_dir, output_dir)
    print(f"✅ Done: {input_dir}\n")
