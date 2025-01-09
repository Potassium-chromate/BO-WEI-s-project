from ultralytics import YOLO

if __name__ == '__main__':
    # Load the YOLOv8 model (pretrained or from scratch)
    model = YOLO('yolov8l.pt')  # Use 'yolov8n.pt', 'yolov8m.pt', or 'yolov8l.pt' based on requirements
    
    # Train the model
    model.train(
        data='data.yaml',      # Path to the dataset configuration file
        epochs=10,             # Number of training epochs
        imgsz=1024,            # Image size for training
        batch=16,              # Batch size
        freeze=10              # Number of layers to freeze for transfer learning
    )
    
    # Validate the model after training
    metrics = model.val()      # Evaluates the model on the validation set

    
    
    
