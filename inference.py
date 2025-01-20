from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


if __name__ == '__main__':
    model = YOLO('runs/detect/train/weights/best.pt')
    image_path = "datasets/test/images/291203_006201.tif"
    ground_truth = "datasets/test/labels/291203_006201.txt"
    image_size = 1000
    
    # Perform inference
    results = model.predict(source = image_path, save=True)
    
    image = Image.open(image_path).convert('RGB')  # Convert the image to RGB
    
    # Plot the image with predictions
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image)
    predict_loc = []
    correct_loc = []
    
    # Accessing the results for plotting bounding boxes
    for result in results:
        # Extract the bounding boxes from the predictions
        boxes = result.boxes.xyxy  # xyxy format for bounding boxes
        
        for box in boxes:
            # Ensure the box is on the CPU and converted to NumPy
            x1, y1, x2, y2 = box.cpu().numpy()  # Convert to NumPy array
            predict_loc.append([x1, y1, x2, y2])
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                              linewidth=2, edgecolor='r', facecolor='none'))
    
    # Remove axes and display the image
    plt.axis('off')
    plt.title("Predicted result")
    plt.savefig('output_image.jpg', bbox_inches='tight', pad_inches=0) 
    plt.show()
    
    # Read the ground thuth location:
    with open(ground_truth, 'r') as f:
        for line in f:
            parts = line.split()
            # Convert the yolo label to absolute coordinate in image
            width = float(parts[3]) * image_size
            height = float(parts[4]) * image_size
            center_x = float(parts[1]) * image_size
            center_y = float(parts[2]) * image_size
            
            # Store in format [x1, y1, x2, y2]
            correct_loc.append([center_x - width / 2,
                                center_y - height / 2,
                                center_x + width / 2,
                                center_y + height / 2])
    
    
    predict_loc = np.array(predict_loc)
    correct_loc = np.array(correct_loc)
    
    # Match the points
    distances = cdist(predict_loc, correct_loc, metric='euclidean')
    matches = []
    unmatched_pred = set(range(len(predict_loc)))
    unmatched_gt = set(range(len(correct_loc)))
    threshold = 15  # Set a threshold for matching
    
    for pred_idx in range(len(predict_loc)):
        gt_idx = np.argmin(distances[pred_idx])
        if distances[pred_idx, gt_idx] < threshold and gt_idx in unmatched_gt:
            matches.append((pred_idx, gt_idx))
            unmatched_pred.discard(pred_idx)
            unmatched_gt.discard(gt_idx)
    
    
    # Plot the matches
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Matching Results")
    
    for pred_idx, gt_idx in matches:
        pred_box = predict_loc[pred_idx]
        gt_box = correct_loc[gt_idx]
        
        # Draw predicted box in green
        x1, y1, x2, y2 = pred_box
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=2, edgecolor='green', facecolor='none'))
        
        # Draw ground truth box in blue
        x1, y1, x2, y2 = gt_box
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=2, edgecolor='blue', linestyle='dashed', facecolor='none'))
    
    plt.axis('off')
    plt.show()
    
    
    # Plot the unmatched points in unmatched_pred
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Unmatching results in predict set")
    
    for pred_idx in unmatched_pred:
        pred_box = predict_loc[pred_idx]
        
        # Draw predicted box in green
        x1, y1, x2, y2 = pred_box
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=2, edgecolor='green', facecolor='none'))
        
    plt.axis('off')
    plt.show()
    
    # Plot the unmatched points in unmatched_gt
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Unmatching results in ground truth set")
    
    for gt_idx in unmatched_gt:
        gt_box = correct_loc[gt_idx]
        
        # Draw ground truth box in blue
        x1, y1, x2, y2 = gt_box
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=2, edgecolor='blue', linestyle='dashed', facecolor='none'))
        
    plt.axis('off')
    plt.show()
