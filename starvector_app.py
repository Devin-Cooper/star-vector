import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from transformers import AutoModelForCausalLM
from starvector.data.util import process_and_rasterize_svg

class StarVectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("StarVector SVG Generator")
        self.root.geometry("800x600")
        
        self.model = None
        self.processor = None
        self.image_path = None
        self.save_dir = os.path.expanduser("~/Documents")
        
        # Create frames
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.preview_frame = tk.Frame(root)
        self.preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.bottom_frame = tk.Frame(root)
        self.bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create buttons
        self.load_model_btn = tk.Button(self.top_frame, text="Load Model", command=self.load_model)
        self.load_model_btn.pack(side=tk.LEFT, padx=5)
        
        self.open_btn = tk.Button(self.top_frame, text="Open Image", command=self.open_image)
        self.open_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_loc_btn = tk.Button(self.top_frame, text="Set Save Location", command=self.set_save_location)
        self.save_loc_btn.pack(side=tk.LEFT, padx=5)
        
        self.generate_btn = tk.Button(self.top_frame, text="Generate SVG", command=self.generate_svg, state=tk.DISABLED)
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        # Create preview canvas
        self.canvas = tk.Canvas(self.preview_frame, bg="lightgray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Status: Ready - Please load model and select an image")
        self.status_label = tk.Label(self.bottom_frame, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X)
        
        # Save path display
        self.save_path_var = tk.StringVar()
        self.save_path_var.set(f"Save location: {self.save_dir}")
        self.save_path_label = tk.Label(self.bottom_frame, textvariable=self.save_path_var, anchor=tk.W)
        self.save_path_label.pack(fill=tk.X, pady=(5, 0))
        
    def load_model(self):
        """Load the StarVector-8B model in full precision"""
        try:
            self.status_var.set("Status: Loading model, please wait...")
            self.root.update_idletasks()
            
            model_name = "starvector/starvector-8b-im2svg"
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            
            # Choose appropriate device
            if torch.cuda.is_available():
                self.device = "cuda"
                self.model.cuda()
            elif torch.backends.mps.is_available():
                self.device = "mps"
                self.model.to(self.device)
            else:
                self.device = "cpu"
                self.status_var.set("Status: WARNING - Using CPU (slow). GPU acceleration not available.")
                
            self.model.eval()
            
            self.processor = self.model.model.processor
            
            self.status_var.set(f"Status: Model loaded successfully (using {self.device})")
            if self.image_path:
                self.generate_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Status: Error loading model")
    
    def open_image(self):
        """Open an image file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        
        if file_path:
            try:
                self.image_path = file_path
                self.display_image(file_path)
                self.status_var.set(f"Status: Image loaded: {os.path.basename(file_path)}")
                
                if self.model is not None:
                    self.generate_btn.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image: {str(e)}")
    
    def set_save_location(self):
        """Set the directory to save the generated SVG"""
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.save_dir = dir_path
            self.save_path_var.set(f"Save location: {self.save_dir}")
    
    def display_image(self, file_path):
        """Display the selected image on the canvas"""
        try:
            img = Image.open(file_path)
            # Resize the image to fit the canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1:  # Canvas not yet rendered
                canvas_width = 600
                canvas_height = 400
            
            img.thumbnail((canvas_width, canvas_height))
            
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.tk_img, anchor='center')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")
    
    def generate_svg(self):
        """Generate SVG from the loaded image using StarVector model"""
        if not self.model or not self.image_path:
            messagebox.showerror("Error", "Please load the model and select an image first")
            return
        
        try:
            self.status_var.set("Status: Generating SVG, please wait...")
            self.generate_btn.config(state=tk.DISABLED)
            self.root.update_idletasks()
            
            # Process the image
            image_pil = Image.open(self.image_path)
            image = self.processor(image_pil, return_tensors="pt")['pixel_values'].to(self.device)
            if not image.shape[0] == 1:
                image = image.squeeze(0)
            batch = {"image": image}
            
            # Generate SVG
            raw_svg = self.model.generate_im2svg(batch, max_length=4000)[0]
            svg, raster_image = process_and_rasterize_svg(raw_svg)
            
            # Save the results
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            svg_path = os.path.join(self.save_dir, f"{base_name}_vector.svg")
            raster_path = os.path.join(self.save_dir, f"{base_name}_raster.png")
            
            # Save SVG
            with open(svg_path, 'w') as f:
                f.write(svg)
            
            # Save rasterized version
            raster_image.save(raster_path)
            
            self.status_var.set(f"Status: SVG generated and saved as {svg_path}")
            messagebox.showinfo("Success", f"SVG generated successfully!\n\nSVG saved as: {svg_path}\nRaster image saved as: {raster_path}")
            
            # Display the generated raster image
            self.display_generated_image(raster_image)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate SVG: {str(e)}")
            self.status_var.set("Status: Error generating SVG")
        finally:
            self.generate_btn.config(state=tk.NORMAL)
    
    def display_generated_image(self, img):
        """Display the generated raster image on the canvas"""
        try:
            # Resize the image to fit the canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1:  # Canvas not yet rendered
                canvas_width = 600
                canvas_height = 400
            
            img.thumbnail((canvas_width, canvas_height))
            
            self.tk_generated_img = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.tk_generated_img, anchor='center')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display generated image: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = StarVectorApp(root)
    root.mainloop()