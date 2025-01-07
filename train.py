# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 23:18:19 2025

@author: user
"""

from ultralytics import YOLO

if __name__ == '__main__':
    
    # Load the YOLOv8 model (pretrained on COCO, or start from scratch)
    model = YOLO('yolov8m.pt')  # Use 'yolov8n' for a small model; adjust to 'yolov8m' or 'yolov8l' as needed.
    
    # Train the model
    model.train(data='data.yaml', epochs=3, imgsz=1000, batch=16)
    
    metrics = model.val()  # Validates the model on the validation set
    print(metrics)
    
    
    