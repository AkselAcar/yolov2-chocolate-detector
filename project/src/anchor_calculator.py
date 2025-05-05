import pandas as pd
import numpy as np
import random

def box_iou(box1, box2):
    """Calculate IoU between two boxes (w, h format)"""
    w1, h1 = box1
    w2, h2 = box2
    intersection = min(w1, w2) * min(h1, h2)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union != 0 else 0.0

def kmeans_plus_plus_init(boxes, k, seed=None):
    """K-means++ initialization with IoU metric"""
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
    """K-means clustering using IoU as distance metric"""
    # Initialization
    centroids = kmeans_plus_plus_init(boxes, k, seed)
    
    for _ in range(max_iter):
        # Assign boxes to nearest centroids
        clusters = [[] for _ in range(k)]
        for box in boxes:
            ious = [box_iou(box, centroid) for centroid in centroids]
            cluster_idx = np.argmax(ious)
            clusters[cluster_idx].append(box)
        
        # Update centroids
        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                # Handle empty clusters by reinitializing
                new_centroids.append(boxes[np.random.randint(len(boxes))])
            else:
                new_centroid = np.mean(cluster, axis=0)
                new_centroids.append(new_centroid)
        new_centroids = np.array(new_centroids)
        
        # Check for convergence
        if np.allclose(centroids, new_centroids, atol=tol):
            break
        centroids = new_centroids
    
    return centroids

# Load and preprocess data
df = pd.read_csv('../labels_YOLO.csv')
original_size = (6000, 4000)
target_size = 640  # Match your input size

# Convert boxes to target size coordinates
scaled_boxes = []
for _, row in df.iterrows():
    w = row['width'] * (target_size / original_size[0])
    h = row['height'] * (target_size / original_size[1])
    scaled_boxes.append([w, h])
scaled_boxes = np.array(scaled_boxes)

# Run k-means clustering
k = 9  # Number of anchors (3 per scale for 3 scales)
anchors = kmeans_iou(scaled_boxes, k=k, seed=42)

# Sort anchors by area
sorted_anchors = sorted(anchors, key=lambda x: x[0] * x[1])

# Calculate average IoU
total_iou = 0.0
for box in scaled_boxes:
    ious = [box_iou(box, anchor) for anchor in sorted_anchors]
    total_iou += max(ious)
average_iou = total_iou / len(scaled_boxes)

# Print results
print(f"Optimal {k} anchors for {target_size}x{target_size} input:")
for i, (w, h) in enumerate(sorted_anchors):
    print(f"Anchor {i+1}: {w:.2f}w x {h:.2f}h (area: {w*h:.2f}px²)")

print(f"\nAverage IoU with closest anchor: {average_iou:.4f}")

# Format for YOLO configuration
yolo_anchors = [f"{int(w)},{int(h)}" for w, h in sorted_anchors]
print("\nYOLO format anchors (width,height):")
print(", ".join(yolo_anchors))