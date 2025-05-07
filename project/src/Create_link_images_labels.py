import os
import csv

image_dir = "../dataset_project_iapr2025/train"
label_dir = "labels"
output_csv = "link_images_labels_YOLO_v3.csv"

images = sorted([f for f in os.listdir(image_dir) if f.endswith('.JPG')])
labels = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_name', 'label_name'])
    for img, lbl in zip(images, labels):
        writer.writerow([img, lbl])