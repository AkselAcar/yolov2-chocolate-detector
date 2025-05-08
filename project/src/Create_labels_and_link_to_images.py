import os
import csv

image_dir = "../dataset_project_iapr2025/train"
label_dir = "labels"
output_csv = "link_images_labels_YOLO_v3.csv"

import pandas as pd
import os

def convert_labels(csv_path, output_dir, img_size=640):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    for _, row in df.iterrows():
        filename, cx, cy, w, h, cls = row
        orig_w, orig_h = 6000, 4000
        
        # Calculate scaling factors
        scale_w = img_size / orig_w # so 640/6000 = 0.10666666666666667
        scale_h = img_size / orig_h # so 640/4000 = 0.16
        
        # Scale coordinates on resized image
        scaled_cx = cx * scale_w
        scaled_cy = cy * scale_h
        scaled_w = w * scale_w
        scaled_h = h * scale_h
        
        # Normalize coordinates to [0, 1] !!!
        norm_cx = scaled_cx / img_size
        norm_cy = scaled_cy / img_size
        norm_w = scaled_w / img_size
        norm_h = scaled_h / img_size
        
        # Write to file : labels/filename.txt => one file per image !!!
        txt_path = os.path.join(output_dir, filename.replace('.JPG', '.txt'))
        with open(txt_path, 'a') as f:
            f.write(f"{cls} {norm_cx} {norm_cy} {norm_w} {norm_h}\n")

convert_labels('../labels_YOLO.csv', 'labels') # ATTENTION TO RUN ONCE PSK SINON CA RAJOUTE DES LIGNES FAUDRAIT LE METTRE DANS UN IF


images = sorted([f for f in os.listdir(image_dir) if f.endswith('.JPG')])
labels = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_name', 'label_name'])
    for img, lbl in zip(images, labels):
        writer.writerow([img, lbl])