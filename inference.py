from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    model = YOLO('runs/detect/train/weights/best.pt')
    image_path = "datasets/test/images/291203_006201.tif"
    ground_truth = "datasets/test/labels/291203_006201.txt"
    
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
            predict_loc.append([x1, y2, x2, y2])
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                              linewidth=2, edgecolor='r', facecolor='none'))
    
    # Remove axes and display the image
    plt.axis('off')
    plt.savefig('output_image.jpg', bbox_inches='tight', pad_inches=0) 
    plt.show()
    
    # Read the ground thuth location:
    with open(ground_truth, 'r') as f:
        for line in f:
            parts = line.split()
            part_float = [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[4])]
            correct_loc.append(part_float)
    
    
    predict_loc = np.array(predict_loc)
    correct_loc = np.array(correct_loc)
    
    predict_loc = np.sort(predict_loc, axis = 0)
    correct_loc = np.sort(correct_loc, axis = 0)
    
