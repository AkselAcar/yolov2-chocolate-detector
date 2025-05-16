import pandas as pd
import numpy as np
import random
import csv
import os
import matplotlib.pyplot as plt

def box_iou(box1, box2):
    w1, h1 = box1
    w2, h2 = box2
    intersection = min(w1, w2) * min(h1, h2)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union != 0 else 0.0

def kmeans_plus_plus_init(boxes, k, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    centroids = [boxes[np.random.randint(len(boxes))]]
    for _ in range(1, k):
        distances = []
        for box in boxes:
            ious = [box_iou(box, centroid) for centroid in centroids]
            distances.append(1 - max(ious))
        probabilities = np.array(distances) / np.sum(distances)
        centroids.append(boxes[np.random.choice(len(boxes), p=probabilities)])
    return np.array(centroids)

def kmeans_iou(boxes, k=9, max_iter=100, tol=1e-4, seed=None):
    centroids = kmeans_plus_plus_init(boxes, k, seed)
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for box in boxes:
            ious = [box_iou(box, centroid) for centroid in centroids]
            cluster_idx = np.argmax(ious)
            clusters[cluster_idx].append(box)
        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                new_centroids.append(boxes[np.random.randint(len(boxes))])
            else:
                new_centroids.append(np.mean(cluster, axis=0))
        new_centroids = np.array(new_centroids)
        if np.allclose(centroids, new_centroids, atol=tol):
            break
        centroids = new_centroids
    return centroids

# Load boxes in pixels directly (no scaling)
df = pd.read_csv('./raw_anotations_YOLO.csv')  # Assumes width and height are in pixels

# List of boxes in (w, h) format in original resolution (6000x4000)
boxes = df[['width', 'height']].values.astype(float)

# Run k-means on original size boxes
k = 9
anchors = kmeans_iou(boxes, k=k, seed=42)

# Sort anchors by area
sorted_anchors = sorted(anchors, key=lambda x: x[0] * x[1])


original_width, original_height = 6000, 4000

# Normalize anchors to [0, 1] range
normalized_anchors = [
    (w / original_width, h / original_height) for (w, h) in sorted_anchors
]

# Average IoU
total_iou = 0.0
for box in boxes:
    ious = [box_iou(box, anchor) for anchor in sorted_anchors]
    total_iou += max(ious)
average_iou = total_iou / len(boxes)

# Output
print(f"Optimal {k} anchors for original resolution (6000x4000):")
for i, (w, h) in enumerate(sorted_anchors):
    print(f"Anchor {i+1}: {w:.2f}w x {h:.2f}h (area: {w*h:.2f}px²)")
print(f"\nAverage IoU with closest anchor: {average_iou:.4f}")

# Save formats
os.makedirs('output_anchors', exist_ok=True)

# 1. Save anchors in pixels format (width, height)
with open('output_anchors/yolo_anchors.yaml', 'w') as f:
    f.write(f"anchors:\n")
    for i in range(0, len(sorted_anchors), 3):
        layer_anchors = sorted_anchors[i:i+3]
        anchors_str = ", ".join([f"[{int(w)}, {int(h)}]" for w, h in layer_anchors])
        f.write(f"  - {anchors_str}\n")

# 4. Save normalized anchors (YOLO format in [0,1])
with open('output_anchors/yolo_anchors_normalized.yaml', 'w') as f:
    f.write("anchors:\n")
    for i in range(0, len(normalized_anchors), 3):
        group = normalized_anchors[i:i+3]
        line = ", ".join([f"[{w:.5f}, {h:.5f}]" for w, h in group])
        f.write(f"  - {line}\n")

# Plot
plt.figure(figsize=(10, 10))
for w, h in boxes[:500]:
    plt.scatter(w, h, color='blue', alpha=0.1, s=10)
for i, (w, h) in enumerate(sorted_anchors):
    plt.scatter(w, h, color='red', s=200, marker='x', linewidths=2)
    plt.text(w, h, f'A{i+1}', fontsize=12, ha='center', va='bottom')
plt.xlabel('Width (pixels)')
plt.ylabel('Height (pixels)')
plt.title('Anchors vs Actual Boxes (6000x4000) — Blue=GT, Red=Anchors')
plt.savefig('output_anchors/anchors_visualization.png')
plt.close()
