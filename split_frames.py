import os
import shutil
import random

# Configuration
input_dirs = {
    'real': 'data/extracted_frames/real',
    'fake': 'data/extracted_frames/fake'
}
output_base = 'data/split_frames'
split_ratios = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}
# Create split directories
for split in split_ratios.keys():
    for label in input_dirs.keys():
        split_dir = os.path.join(output_base, split, label)
        os.makedirs(split_dir, exist_ok=True)

# Split and move files
for label, input_dir in input_dirs.items():
    images = [img for img in os.listdir(input_dir) if img.endswith('.jpg')]
    random.shuffle(images)

    total = len(images)
    train_end = int(split_ratios['train'] * total)
    val_end = train_end + int(split_ratios['val'] * total)

    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split, files in splits.items():
        for img in files:
            src = os.path.join(input_dir, img)
            dst = os.path.join(output_base, split, label, img)
            shutil.copy2(src, dst)

print("Dataset split complete.")
