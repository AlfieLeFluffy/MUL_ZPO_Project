import tkinter as tk
from src.image import CImage


class StatusBar:
    def __init__(self):
        self.setup_status_bar()

    def setup_status_bar(self):
        """Initial setup for a status bar"""
        self.statusbar = tk.Frame(padx=4, pady=4)
        self.resolution_label = tk.Label(self.statusbar, text="Resolution: ")
        self.resolution_label.pack(anchor="w")
        self.statusbar.pack(side="bottom", fill="x")

    def update_status_bar(self, _image: CImage):
        """Updates the status bar with image information.

        Args:
            _image (CImage): Image containing the information
        """
        self.resolution_label.destroy()
        self.resolution_label = tk.Label(
            self.statusbar, text=f"Resolution: {_image.size[0]} x {_image.size[1]} "
        )
        self.resolution_label.pack(anchor="w")

    def reset_status_bar(self):
        """Resets the status bar to an empty state."""
        self.resolution_label.destroy()
        self.resolution_label = tk.Label(self.statusbar, text="Resolution: ")
        self.resolution_label.pack(anchor="w")
