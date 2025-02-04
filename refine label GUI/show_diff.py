from ultralytics import YOLO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from tkinter import Tk, filedialog

class Show_diff:
    def __init__(self):
        self.model = None
        self.img_size = 1000
        self.source_img = None # opened by PIL library
        
        self.source_img_path = None
        self.source_model_path = None
        self.source_label_path = None        

        self.algo_diff_image = None
        self.model_diff_image = None
        
        self.model_label = []
        self.algo_label = []
        self.unmatched_model_idx = []
        self.unmatched_algo_idx =[]
        self.model_save = {}
        self.algo_save = {}
        self.unmatched_model_highlight_idx = set()
        self.unmatched_algo_highlight_idx = set()
        
        
        """Temp variable"""
        self.source_img_tmp = None
        self.algo_label_tmp = []
    
    def flush(self):
        self.source_img = self.source_img_tmp.copy()
        self.algo_label = self.algo_label_tmp.copy()
        self.source_img_tmp = None
        self.algo_label_tmp.clear()
        self.unmatched_model_idx.clear()
        self.unmatched_algo_idx.clear()
        
    def select_image(self):
        """Function to handle image selection and loading."""
        # Select image file
        self.source_img_path = self.select_path("Select Image")
        
        if self.source_img_path:  # Ensure a valid file was selected
            # Open the image using PIL
            self.source_img_tmp = Image.open(self.source_img_path).convert('RGB')
        else:
            print("No file selected.")
    
    def select_model(self):
        """Function to handle model selection and loading."""
        # Select image file
        self.source_model_path = self.select_path("Select Model")
        
        if self.source_model_path:  # Ensure a valid file was selected
            # Open the image using PIL
            self.model = YOLO(self.source_model_path)
        else:
            print("No model selected.")
    
    def select_algo_label(self):
        """Function to handle algo label selection and loading."""
        # Select image file
        self.source_label_path = self.select_path("Select Algorithm Label")
        
        if self.source_label_path:  # Ensure a valid file was selected
            self.algo_label_tmp.clear()
            # Read the ground thuth location:
            with open(self.source_label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    # Convert the yolo label to absolute coordinate in image
                    width = float(parts[3]) * self.img_size
                    height = float(parts[4]) * self.img_size
                    center_x = float(parts[1]) * self.img_size
                    center_y = float(parts[2]) * self.img_size
                    
                    # Store in format [x1, y1, x2, y2]
                    self.algo_label_tmp.append([center_x - width / 2,
                                        center_y - height / 2,
                                        center_x + width / 2,
                                        center_y + height / 2])
        else:
            print("No algorithm label selected.")
    
    def select_path(self, title):
        """Open the file dialog and return the selected file path."""
        file_path = filedialog.askopenfilename(title=title)
        return file_path  # Return the file path or None if no file was selected
     
    def predict(self):
        if not self.model:
            print("The model is not loaded yet!")
            return False
        # Perform inference
        results = self.model.predict(source = self.source_img_path, save=False)
        self.model_label.clear()
        # Accessing the results for plotting bounding boxes
        for result in results:
            # Extract the bounding boxes from the predictions
            boxes = result.boxes.xyxy  # xyxy format for bounding boxes
            
            for box in boxes:
                # Ensure the box is on the CPU and converted to NumPy
                x1, y1, x2, y2 = box.cpu().numpy()  # Convert to NumPy array
                self.model_label.append([x1, y1, x2, y2])
        return True
    
    def match(self):
        #if not self.model_label or not self.algo_label:
        if not self.model_label or not self.algo_label:
            print("The label is not load correctly!")
            return False
        model_loc = np.array(self.model_label)
        algo_loc = np.array(self.algo_label)
        
        # Match the points
        distances = cdist(model_loc, algo_loc, metric='euclidean')
        unmatched_model = set(range(len(model_loc)))
        unmatched_algo = set(range(len(algo_loc)))
        threshold = 15  # Set a threshold for matching
        
        for model_idx in range(len(model_loc)):
            algo_idx = np.argmin(distances[model_idx])
            if distances[model_idx, algo_idx] < threshold and algo_idx in unmatched_algo:
                unmatched_model.discard(model_idx)
                unmatched_algo.discard(algo_idx)
        
        self.unmatched_model_idx = list(unmatched_model)
        self.unmatched_algo_idx = list(unmatched_algo)
        self.unmatched_model_highlight_idx.clear()
        self.unmatched_algo_highlight_idx.clear()
        return True
    
    def make_graph(self):
        if not self.source_img:
            print("No image loaded.")
            return False
        
        # Create a copy of the source image to avoid modifying the original
        self.model_diff_image = self.source_img.copy()
        self.algo_diff_image = self.source_img.copy()
    
        # Create a drawing context
        draw_model = ImageDraw.Draw(self.model_diff_image)
        draw_algo = ImageDraw.Draw(self.algo_diff_image)

        # Draw the unmatched model bounding boxes in GREEN
        for model_idx in self.unmatched_model_idx:
            x1, y1, x2, y2 = self.model_label[model_idx]
            if model_idx in self.unmatched_model_highlight_idx:
                draw_model.rectangle([x1, y1, x2, y2], outline="yellow", width=2)
            else:
                draw_model.rectangle([x1, y1, x2, y2], outline="green", width=2)
        
        # Draw the unmatched algo bounding boxes in BLUE
        for algo_idx in self.unmatched_algo_idx:
            x1, y1, x2, y2 = self.algo_label[algo_idx]
            if algo_idx in self.unmatched_algo_highlight_idx:
                draw_algo.rectangle([x1, y1, x2, y2], outline="yellow", width=2)
            else:
                draw_algo.rectangle([x1, y1, x2, y2], outline="blue", width=2)
        
        return True
    
    def save_new_label(self):
        if not self.algo_label:
            print("No algo labels exist")
            return
        if not self.model_label:
            print("No model labels exist")
            return
        
        save_label = []
        
        # Save the points that the algorithm found correctly
        for i in range(len(self.algo_label)):
            if i in self.algo_save and not self.algo_save[i]:
                continue
            save_label.append(self.algo_label[i])
        
        # Save the rest of the points that the model found but the algorithm didn't
        for i in range(len(self.model_label)):
            if i in self.model_save and self.model_save[i]:
                save_label.append(self.model_label[i])
        
        # The stored format in save_label is [x1, y1, x2, y2]
        # Turn it into [0, x_center, y_center, width, height]
        for i in range(len(save_label)):
            width = (save_label[i][2] - save_label[i][0]) / self.img_size
            height = (save_label[i][3] - save_label[i][1]) / self.img_size
            x_center = (save_label[i][2] + save_label[i][0]) / self.img_size / 2
            y_center = (save_label[i][3] + save_label[i][1]) / self.img_size / 2
            save_label[i] = [0, x_center, y_center, width, height]
            
        # Open file dialog to select where to save the file
        save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        
        if save_path:
            try:
                with open(save_path, 'w') as f:
                    for label in save_label:
                        # Format the label data for YOLO (for example: x_center, y_center, width, height, class_id)
                        # Assuming label is a list or tuple with [class_id, x_center, y_center, width, height]
                        # Modify this format according to your label data
                        label_str = ' '.join([f"{val:.4f}" for val in label])  # Convert list to space-separated string
                        f.write(f"{label_str}\n")
                print(f"Labels saved successfully to {save_path}")
            except Exception as e:
                print(f"Error saving file: {e}")
        
        
        
if __name__ == '__main__':
    show_diff = Show_diff()
    model_path = show_diff.select_path("model path")
    show_diff.model = YOLO(model_path)