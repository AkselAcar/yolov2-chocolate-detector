import os
import csv
import glob
import pandas as pd


image_dir = "../../dataset_project_iapr2025/train"
label_dir = "labels"
output_csv = "link_images_labels_YOLO.csv"


def convert_labels(csv_path, output_dir, img_width=576, img_height=384):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    for _, row in df.iterrows():
        filename, cx, cy, w, h, cls = row
        orig_w, orig_h = 6000, 4000
        
        # Calculate scaling factors
        scale_w = img_width / orig_w # so 576/6000 = 0.096
        scale_h = img_height / orig_h # so 384/4000 = 0.096
        
        # Scale coordinates on resized image
        scaled_cx = cx * scale_w
        scaled_cy = cy * scale_h
        scaled_w = w * scale_w
        scaled_h = h * scale_h
        
        # Normalize coordinates to [0, 1] !!!
        norm_cx = scaled_cx / img_width
        norm_cy = scaled_cy / img_height
        norm_w = scaled_w / img_width
        norm_h = scaled_h / img_height
        
        # Write to file : labels/filename.txt => one file per image
        txt_path = os.path.join(output_dir, filename.replace('.JPG', '.txt'))
        with open(txt_path, 'a') as f:
            f.write(f"{cls} {norm_cx} {norm_cy} {norm_w} {norm_h}\n")


if os.path.exists(label_dir):
    # Remove existing labels directory
    for file_path in glob.glob(os.path.join(label_dir, "*.txt")):
        os.remove(file_path)

convert_labels('raw_annotations_YOLO.csv', 'labels')

# Create a CSV file linking images to their labels
images = sorted([f for f in os.listdir(image_dir) if f.endswith('.JPG')])
labels = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_name', 'label_name'])
    for img, lbl in zip(images, labels):
        writer.writerow([img, lbl])