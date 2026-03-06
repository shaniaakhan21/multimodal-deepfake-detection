import cv2
import os

def extract_frames_from_videos(input_dir, output_dir, label, fps_interval=1):
    os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        vidcap = cv2.VideoCapture(video_path)

        frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(frame_rate * fps_interval)

        count = 0
        saved = 0
        success, image = vidcap.read()
        while success:
            if count % frame_interval == 0:
                frame_filename = f"{label}_{video_file[:-4]}_frame{saved}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, image)
                saved += 1
            success, image = vidcap.read()
            count += 1

        print(f"Extracted {saved} frames from {video_file}")

# Paths
real_video_dir = "data/faceforensics/original_sequences/youtube/c23/videos"
fake_video_dir = "data/faceforensics/manipulated_sequences/DeepFakeDetection/c23/videos"
output_real = "data/extracted_frames/real"
output_fake = "data/extracted_frames/fake"

# Extract frames
extract_frames_from_videos(real_video_dir, output_real, label='real')
extract_frames_from_videos(fake_video_dir, output_fake, label='fake')



