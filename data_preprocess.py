import os
import cv2
from tqdm import tqdm
from pathlib import Path
import argparse
import gc

def process_and_save(input_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for label in ['real', 'fake']:
        src_folder = os.path.join(input_dir, label)
        dst_folder = os.path.join(output_dir, label)
        Path(dst_folder).mkdir(parents=True, exist_ok=True)

        existing_files = set(os.listdir(dst_folder))  # Skip already done

        files = os.listdir(src_folder)
        total = len(files)

        print(f"\n🚀 Preprocessing {label.upper()} ({total} images)...")

        for img_name in tqdm(files, desc=f"{label.upper()}", ncols=100):
            if img_name in existing_files:
                continue  # Skip already processed

            img_path = os.path.join(src_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            resized = cv2.resize(img, (224, 224))
            save_path = os.path.join(dst_folder, img_name)
            cv2.imwrite(save_path, resized)

def main():
    parser = argparse.ArgumentParser(description="Preprocess split frame images")
    parser.add_argument('--mode', choices=['train', 'val', 'all'], default='all', help='Which data split to preprocess')
    args = parser.parse_args()

    if args.mode in ['train', 'all']:
        print("\n==============================")
        print("⚙️  Preprocessing training data")
        print("==============================")
        process_and_save('data/cropped/train', 'data/processed/train')

        # Clean memory
        gc.collect()

    if args.mode in ['val', 'all']:
        print("\n==============================")
        print("⚙️  Preprocessing validation data")
        print("==============================")
        process_and_save('data/cropped/val', 'data/processed/val')

        gc.collect()

if __name__ == "__main__":
    main()
