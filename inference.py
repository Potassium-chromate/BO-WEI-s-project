# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 00:03:26 2025

@author: user
"""

from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model = YOLO('runs/detect/train15/weights/best.pt')
    
    # Perform inference
    results = model.predict(source='沉沉291203/291203_006001.tif', save=True)
    
    # Get the image with detections
    image_path = '沉沉291203/291203_006001.tif'
    image = Image.open(image_path).convert('RGB')  # Convert the image to RGB
    
    # Plot the image with predictions
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Accessing the results for plotting bounding boxes
    for result in results:
        # Extract the bounding boxes from the predictions
        boxes = result.boxes.xyxy  # xyxy format for bounding boxes
        
        for box in boxes:
            # Ensure the box is on the CPU and converted to NumPy
            x1, y1, x2, y2 = box.cpu().numpy()  # Convert to NumPy array
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                              linewidth=2, edgecolor='r', facecolor='none'))
    
    # Remove axes and display the image
    plt.axis('off')
    plt.savefig('output_image.jpg', bbox_inches='tight', pad_inches=0) 
    plt.show()