import customtkinter as ctk
from src.image import CImage


class StatusBar:
    def __init__(self, _root):
        self.padding_x = 10
        self.padding_y = 4
        self.setup_status_bar(_root)

    def setup_status_bar(self, _root):
        """Initial setup for a status bar"""
        self.root = _root
        self.statusbar = ctk.CTkFrame(_root, fg_color="gray10")
        self.resolution_label = ctk.CTkLabel(self.statusbar, text="No image loaded")
        self.resolution_label.grid(
            row=0, column=0, padx=self.padding_x, pady=self.padding_y, sticky="e"
        )
        self.statusbar.pack(side="bottom", fill="x")

        self.saved_status_label = ctk.CTkLabel(self.statusbar, text="")
        self.saved_status_label.grid(
            row=0, column=1, padx=self.padding_x, pady=self.padding_y, sticky="e"
        )

    def update_status_bar(self, _image: CImage):
        """Updates the status bar with image information.

        Args:
            _image (CImage): Image containing the information
        """
        if not isinstance(_image, CImage):
            self.reset_status_bar()
            return

        self.resolution_label.destroy()
        self.resolution_label = ctk.CTkLabel(
            self.statusbar, text=f"Resolution: {_image.size[0]} x {_image.size[1]} "
        )
        self.resolution_label.grid(
            row=0, column=0, padx=self.padding_x, pady=self.padding_y, sticky="e"
        )

        self.saved_status_label = ctk.CTkLabel(
            self.statusbar, text=f"Saved: {_image.is_saved()}"
        )
        self.saved_status_label.grid(
            row=0, column=1, padx=self.padding_x, pady=self.padding_y, sticky="e"
        )

    def reset_status_bar(self):
        """Resets the status bar to an empty state."""
        self.resolution_label.destroy()
        self.saved_status_label.destroy()
        self.resolution_label = ctk.CTkLabel(self.statusbar, text="No image loaded")
        self.resolution_label.grid(
            row=0, column=0, padx=self.padding_x, pady=self.padding_y, sticky="e"
        )
