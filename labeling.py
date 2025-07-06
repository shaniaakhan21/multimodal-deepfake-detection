import pandas as pd
import os

def create_labels_csv(base_path, output_csv='data/labels.csv'):
    data = {'filepath': [], 'label': []}
    for split in ['train', 'val']:
        for label in ['real', 'fake']:
            folder = os.path.join(base_path, split, label)
            for img in os.listdir(folder):
                img_path = os.path.join(folder, img)
                data['filepath'].append(img_path)
                data['label'].append(0 if label == 'real' else 1)
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✅ Labels saved to {output_csv}")

# Example usage
create_labels_csv('data/cropped')
