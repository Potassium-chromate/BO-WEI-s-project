# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 21:21:13 2025

@author: user
"""

import scipy.io
import os

# Define paths
image_folder = "./沉沉291203"  # Folder containing images
mat_folder = "./mat"         # Folder containing .mat files
label_folder = "./labels"    # Folder to save YOLO label files
os.makedirs(label_folder, exist_ok=True)

w = 1000  # Width of the images
h = 1000  # Height of the images

counter = 0
limit = 300
for mat_file in os.listdir(mat_folder):
    if counter > limit - 1:
         break
    mat_path = os.path.join(mat_folder, mat_file)
    data = scipy.io.loadmat(mat_path)
    counter += 1

    # Get circle data
    circles = data['CandA']  # Assuming 'CandA' contains (x, y, r) for circles
    
    # Path for the corresponding label file
    label_path = os.path.join(label_folder, mat_file.replace(".mat", ".txt"))
    
    # Open label file for writing (all circles in one file)
    with open(label_path, "w") as label_file:
        for c in circles:
            x, y, r = c[0], c[1], c[2]
            norm_cx = x / w
            norm_cy = y / h
            norm_width = (2 * r) / w
            norm_height = (2 * r) / h

            # Write circle info in YOLO format
            label_file.write(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_width:.6f} {norm_height:.6f}\n")

print("YOLO labels generated successfully.")
