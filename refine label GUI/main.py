import tkinter as tk
from PIL import Image, ImageTk
from tkinter import Canvas, Scrollbar, Frame, Tk
from show_diff import Show_diff

class LabelRefinerApp(tk.Tk):
    def __init__(self, lib):
        super().__init__()
        self.lib = lib
        
        # Window properties
        self.title("Label Refiner")
        self.geometry("1920x1080")
     
        # Initialize UI components inside the scrollable frame
        self.algo_checkbox_vars = []
        self.model_checkbox_vars = []
        
        self.create_button()
        self.image_name_label, self.model_name_label, self.label_name_label = self.create_text()
        self.algo_image_label, self.model_image_label = self.create_image_display()
        self.get_checkbox_states()
      
    def create_button(self):
        button_frame = Frame(self)
        button_frame.place(x=10, y=10)
        """Create button for selecting image."""
        # Button to select image
        button = tk.Button(button_frame, text="Select Image", command=self.select_image, font=("Arial", 16))
        button.pack(pady=10, fill="x")
    
        """Create button for selecting model."""
        button = tk.Button(button_frame, text="Select Model", command=self.select_model, font=("Arial", 16))
        button.pack(pady=10, fill="x")
        
        """Create button for selecting model."""
        button = tk.Button(button_frame, text="Select Algo Label", command=self.select_label, font=("Arial", 16))
        button.pack(pady=10, fill="x")
        
        """Create button for showing difference."""
        button = tk.Button(button_frame, text="Show Difference", command=self.show_difference, font=("Arial", 16))
        button.pack(pady=10, fill="x")
        
        """Create button for save new label in text file."""
        button = tk.Button(self, text="Save Label", command=self.save_new_label, font=("Arial", 16))
        button.pack(pady=10, anchor="ne")
        
        
    def create_text(self):
        text_frame = Frame(self)
        text_frame.place(x=250, y=10)
        """Create the label to display text."""
        image_name_label = tk.Label(text_frame, text="Image name: ", font=("Arial", 12))  # To display image name
        image_name_label.pack(pady=15, anchor="w")
        
        model_name_label = tk.Label(text_frame, text="Model source: ", font=("Arial", 12))  # To display image name
        model_name_label.pack(pady=15, anchor="w")
        
        label_name_label = tk.Label(text_frame, text="Label source: ", font=("Arial", 12))  # To display image name
        label_name_label.pack(pady=15, anchor="w")
        
        return image_name_label, model_name_label, label_name_label
    
    def create_image_display(self):
        """Create the label to display the difference in algo."""
        algo_image_frame = Frame(self)
        algo_image_frame.place(x=400, y=250)
        #Use label to display image
        algo_image_label = tk.Label(algo_image_frame)
        algo_image_label.pack(side="bottom", fill="x")
        #Place the title
        algo_title_label = tk.Label(algo_image_frame, text="Algo Difference", font=("Arial", 18))  # To display image path
        algo_title_label.pack(side="top", fill="x")


        """Create the label to display the difference in model."""
        model_image_frame = Frame(self)
        model_image_frame.place(x=1200, y=250)
        model_image_label = tk.Label(model_image_frame)  # To display image name
        model_image_label.pack(side="bottom", fill="x")
        model_title_label = tk.Label(model_image_frame, text="Model Difference", font=("Arial", 18))  # To display image path
        model_title_label.pack(side="top", fill="x")
        return algo_image_label, model_image_label

    def update_name_display(self):
        """Update all the text."""
        #Update image path
        if self.lib.source_img_path:
            name = self.lib.source_img_path.split("/")[-1]
            text = "Image name: " + name
            self.image_name_label.config(text=text)
        #Update mode path
        if self.lib.source_model_path:
            name = self.lib.source_model_path.split("/")[-1]
            text = "Model source: " + name
            self.model_name_label.config(text=text)
        #Update label path
        if self.lib.source_label_path:
            name = self.lib.source_label_path.split("/")[-1]
            text = "Label source: " + name
            self.label_name_label.config(text=text)
       
    
    def update_image_display(self):
        """Update the image label with the selected image."""
        if self.lib.algo_diff_image:
            # Resize image to 500x500
            resized_img = self.lib.algo_diff_image.resize((700, 700))
            # Convert the image to a Tkinter-compatible format
            img_tk = ImageTk.PhotoImage(resized_img)
            self.algo_image_label.config(image=img_tk)
            self.algo_image_label.image = img_tk  # Keep a reference to the image
        
        if self.lib.model_diff_image:
            # Resize image to 500x500
            resized_img = self.lib.model_diff_image.resize((700, 700))
            # Convert the image to a Tkinter-compatible format
            img_tk = ImageTk.PhotoImage(resized_img)
            self.model_image_label.config(image=img_tk)
            self.model_image_label.image = img_tk  # Keep a reference to the image

    def create_checkbox_list(self):
        """Create a list where each element has 'Save' and 'Show' checkboxes."""
        # Destroy old frames if they exist
        if hasattr(self, 'algo_checkbox_frame'):
            self.algo_checkbox_frame.destroy()
        if hasattr(self, 'model_checkbox_frame'):
            self.model_checkbox_frame.destroy()
            
        # Create main frame
        algo_diff_frame = Frame(self)
        algo_diff_frame.place(x=10, y=260)
        model_diff_frame = Frame(self)
        model_diff_frame.place(x=10, y=650)
        
        # Place the title
        algo_fram_title = tk.Label(algo_diff_frame, text="Algo Difference: ", font=("Arial", 12))
        algo_fram_title.pack(side="top", fill="x")
        model_fram_title = tk.Label(model_diff_frame, text="Model Difference: ", font=("Arial", 12))
        model_fram_title.pack(side="top", fill="x")


        # Create canvas to hold checkbuttons
        algo_diff_canvas = Canvas(algo_diff_frame, width=300, height=300)
        algo_diff_canvas.pack(side="left", fill="y")
        model_diff_canvas = Canvas(model_diff_frame, width=300, height=300)
        model_diff_canvas.pack(side="left", fill="y")
        
        # Place the Scrollbar to the right side of the frame
        algo_scrollbar = Scrollbar(algo_diff_frame, orient="vertical", command=algo_diff_canvas.yview)
        algo_scrollbar.pack(side="right", fill="y")
        algo_diff_canvas.config(yscrollcommand=algo_scrollbar.set)
        
        model_scrollbar = Scrollbar(model_diff_frame, orient="vertical", command=model_diff_canvas.yview)
        model_scrollbar.pack(side="right", fill="y")
        model_diff_canvas.config(yscrollcommand=model_scrollbar.set)

        # Create inner frames that will hold checkboxes
        self.algo_checkbox_frame = Frame(algo_diff_canvas)
        self.model_checkbox_frame = Frame(model_diff_canvas)
    
        algo_diff_canvas.create_window((0, 0), window=self.algo_checkbox_frame, anchor="nw")
        model_diff_canvas.create_window((0, 0), window=self.model_checkbox_frame, anchor="nw")

        # Clear old checkbox variables
        self.algo_checkbox_vars.clear() 
        self.model_checkbox_vars.clear() 
        
        #Show the points that are unmatched in algorithm
        if self.lib.unmatched_algo_idx and self.lib.algo_label:
            for algo_idx in self.lib.unmatched_algo_idx:
                row_frame = tk.Frame(self.algo_checkbox_frame)
                row_frame.pack(fill="x", anchor="w")
                
                row_frame.grid_columnconfigure(0, weight=1, uniform="group1")  
                row_frame.grid_columnconfigure(1, weight=0)
                row_frame.grid_columnconfigure(2, weight=0)
                
                text = str([int(round(x)) for x in self.lib.algo_label[algo_idx]])
                label = tk.Label(row_frame, text=text, font=("Arial", 10))
                label.grid(row=0, column=0, sticky="w", padx=5)
    
                var_save = tk.BooleanVar()
                var_show = tk.BooleanVar()
                self.algo_checkbox_vars.append((var_save, var_show))
    
                save_cb = tk.Checkbutton(row_frame, text="Save", variable=var_save)
                save_cb.grid(row=0, column=1, sticky="e", padx=5)
    
                show_cb = tk.Checkbutton(row_frame, text="Show", variable=var_show)
                show_cb.grid(row=0, column=2, sticky="e", padx=5)
        
        #Show the points that are unmatched in model
        if self.lib.unmatched_model_idx and self.lib.model_label:
            for model_idx in self.lib.unmatched_model_idx:
                row_frame = tk.Frame(self.model_checkbox_frame)
                row_frame.pack(fill="x", anchor="w")
                
                row_frame.grid_columnconfigure(0, weight=1, uniform="group1")  
                row_frame.grid_columnconfigure(1, weight=0)
                row_frame.grid_columnconfigure(2, weight=0)
                
                text = str([int(round(x)) for x in self.lib.model_label[model_idx]])
                label = tk.Label(row_frame, text=text, font=("Arial", 10))
                label.grid(row=0, column=0, sticky="w", padx=5)
    
                var_save = tk.BooleanVar()
                var_show = tk.BooleanVar()
                self.model_checkbox_vars.append((var_save, var_show))
    
                save_cb = tk.Checkbutton(row_frame, text="Save", variable=var_save)
                save_cb.grid(row=0, column=1, sticky="e", padx=5)
    
                show_cb = tk.Checkbutton(row_frame, text="Show", variable=var_show)
                show_cb.grid(row=0, column=2, sticky="e", padx=5)
                
        self.algo_checkbox_frame.update_idletasks()
        algo_diff_canvas.config(scrollregion=algo_diff_canvas.bbox("all"))
        self.model_checkbox_frame.update_idletasks()
        model_diff_canvas.config(scrollregion=model_diff_canvas.bbox("all"))
                
    def get_checkbox_states(self):
        """Retrieve the states of checkboxes."""
        if self.lib.source_img and self.lib.unmatched_algo_idx:
            #Deal with the checkbox in unmatched algo part
            for check_box_idx, algo_idx in enumerate(self.lib.unmatched_algo_idx):
                #Get the boolean in "show"
                if self.algo_checkbox_vars[check_box_idx][1].get():
                    self.lib.unmatched_algo_highlight_idx.add(algo_idx)     
                else:
                    self.lib.unmatched_algo_highlight_idx.discard(algo_idx)
                    
                #Get the boolean in "save"
                self.lib.algo_save[algo_idx] = self.algo_checkbox_vars[check_box_idx][0].get()
                    
        if self.lib.source_img and self.lib.unmatched_model_idx:          
            #Deal with the checkbox in unmatched model part
            for check_box_idx, model_idx in enumerate(self.lib.unmatched_model_idx):
                #Get the boolean in "show"
                if self.model_checkbox_vars[check_box_idx][1].get():
                    self.lib.unmatched_model_highlight_idx.add(model_idx)        
                else:
                    self.lib.unmatched_model_highlight_idx.discard(model_idx)
                
                #Get the boolean in "save"
                self.lib.model_save[model_idx] = self.model_checkbox_vars[check_box_idx][0].get()
                    
            self.lib.make_graph()
            self.update_image_display()
        # Call this function again after 500ms
        self.after(200, self.get_checkbox_states)
    def select_image(self):
        self.lib.select_image()
        self.update_name_display()
    
    def select_model(self):
        self.lib.select_model()
        self.update_name_display()
        
    def select_label(self):
        self.lib.select_algo_label()
        self.update_name_display()
        
    def show_difference(self):
        self.lib.flush()
        if not self.lib.predict():
            return
        if not self.lib.match():
            return
        if not self.lib.make_graph():
            return
        self.update_image_display()
        self.create_checkbox_list()
        
    def save_new_label(self):
        self.get_checkbox_states()
        self.lib.save_new_label()
        self.lib.model_save.clear()
        self.lib.algo_save.clear()
        
if __name__ == "__main__":
    lib = Show_diff()
    app = LabelRefinerApp(lib)
    app.mainloop()
