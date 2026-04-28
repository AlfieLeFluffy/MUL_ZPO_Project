import tkinter as tk
from src.image import CImage
from PIL import Image, ImageTk

class Tab:

    MAIN_TAB: str = "Main"

    def __init__(self, _tab_control, _image: CImage = None, _saved: bool = False, _menu_tab: bool = False, _menu_controls: dict = None):
        self.tab_control = _tab_control
        self.menu_tab = _menu_tab
        self.menu_controls = _menu_controls
        self.image = _image
        self.saved = _saved

        if self.menu_tab:
            self.setup_menu_tab()
        elif not self.menu_tab and self.image:
            self.setup_image_tab()
    
    def setup_menu_tab(self):
        self.saved = True
        self.tab_frame = tk.Frame(self.tab_control, bg="white")
        for key in self.menu_controls.keys():
            button = tk.Button(self.tab_frame, text=key, command=self.menu_controls[key], padx=12, pady=4)
            button.pack(anchor="nw", padx=10, pady=10)
        self.tab_control.add(self.tab_frame, text=self.MAIN_TAB, padding=10)
        self.tab_control.pack(expand = 1, fill ="both")
        return self

    def setup_image_tab(self):
        label_name: str = self.image.name
        self.tab_frame = tk.Frame(self.tab_control)
        self.tab_control.add(self.tab_frame, text=label_name, padding=10)
        self.tab_control.pack(expand = 1, fill ="both")
        cimage: CImage = self.image
        
        if cimage.type == CImage.IMAGE_TYPE.RGB_INT:
            data = cimage.data

        img = Image.fromarray(data)
        photo_img = ImageTk.PhotoImage(image=img)

        img_label = tk.Label(self.tab_frame, image=photo_img)
        img_label.image = photo_img
        img_label.pack()
        return self

