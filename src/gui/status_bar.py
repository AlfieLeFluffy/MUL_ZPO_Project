import tkinter as tk
from src.image import CImage

class StatusBar:
    def __init__(self):
        self.setup_status_bar()

    def setup_status_bar(self):
        self.statusbar = tk.Frame(padx=4, pady=4)
        self.resolution_label = tk.Label(self.statusbar,text=f"Resolution: ")
        self.resolution_label.pack(anchor="w")
        self.statusbar.pack(side="bottom", fill="x")

    def update_status_bar(self, _image: CImage):
        self.resolution_label.destroy()
        self.resolution_label = tk.Label(self.statusbar,text=f"Resolution: {_image.size[0]} x {_image.size[1]} ")
        self.resolution_label.pack(anchor="w")
    
    def reset_status_bar(self):
        self.resolution_label.destroy()
        self.resolution_label = tk.Label(self.statusbar,text=f"Resolution: ")
        self.resolution_label.pack(anchor="w")