import tkinter as tk
import numpy as np
from enum import Enum
from PIL import Image, ImageTk

from src.image import CImage


class Tab:
    MAIN_TAB: str = "Main"

    class TabType(Enum):
        MENU = 1
        IMAGE = 2
        IMAGEKERNEL = 3
        KERNEL = 4

    def __init__(
        self,
        _tab_control,
        _image: CImage = None,
        _saved: bool = False,
        _kernel: np.ndarray = None,
        _menu_tab: bool = False,
        _menu_controls: dict = None,
    ):
        """Initial function for a Tab.

        Args:
            _tab_control (_type_): Main Tab Control
            _image (CImage, optional): Image. Defaults to None.
            _saved (bool, optional): Is the image saved. Defaults to False.
            _menu_tab (bool, optional): Is it a menu tab. Defaults to False.
            _menu_controls (dict, optional): What are the menu controls. Defaults to None.
        """
        self.tab_control = _tab_control
        self.type = self.TabType.MENU if _menu_tab else self.TabType.IMAGE
        self.menu_controls = _menu_controls
        self.image = _image
        self.saved = _saved
        self.kenrel = _kernel

        if self.type == self.TabType.MENU and self.menu_controls is None:
            raise Exception("Menu tab must have menu controls.")

        if (
            self.type == self.TabType.IMAGE
            and self.image is None
            and self.kenrel is None
        ):
            self.type = self.TabType.KERNEL

        if self.type == self.TabType.IMAGE and self.image is None:
            raise Exception("Image tab must have an image.")

        if self.type == self.TabType.MENU and self.image is not None:
            raise Exception("Menu tab cannot have an image.")

        if self.type == self.TabType.IMAGE and self.menu_controls is not None:
            raise Exception("Image tab cannot have menu controls.")

        if self.type == self.TabType.IMAGE and self.kenrel is not None:
            self.type = self.TabType.IMAGEKERNEL

        match self.type:
            case self.TabType.MENU:
                self.setup_menu_tab()
            case self.TabType.IMAGE:
                self.setup_image_tab()
            case self.TabType.IMAGEKERNEL:
                self.setup_image_kernel_tab()
            case self.TabType.KERNEL:
                self.setup_kernel_tab()

    def setup_menu_tab(self):
        """Setup for a menu tab"""
        self.saved = True
        self.tab_frame = tk.Frame(self.tab_control, bg="white")
        for key in self.menu_controls.keys():
            button = tk.Button(
                self.tab_frame,
                text=key,
                command=self.menu_controls[key],
                padx=12,
                pady=4,
            )
            button.pack(anchor="nw", padx=10, pady=10)
        self.tab_control.add(self.tab_frame, text=self.MAIN_TAB, padding=10)
        self.tab_control.pack(expand=1, fill="both")
        return self

    def setup_image_tab(self):
        """Setup for image tab."""
        label_name: str = self.image.name
        self.tab_frame = tk.Frame(self.tab_control)
        self.tab_control.add(self.tab_frame, text=label_name, padding=10)
        self.tab_control.pack(expand=1, fill="both")
        cimage: CImage = self.image

        if cimage.type in [CImage.IMAGE_TYPE.RGB_INT, CImage.IMAGE_TYPE.RGB_DOUBLE]:
            data = cimage.data
        else:
            data = cimage.ycbcr2rgb().data.astype(np.uint8)

        img = Image.fromarray(data)
        photo_img = ImageTk.PhotoImage(image=img)

        img_label = tk.Label(self.tab_frame, image=photo_img)
        img_label.image = photo_img
        img_label.pack()
        return self

    def setup_image_kernel_tab(self):
        """Setup for image kernel tab."""
        label_name: str = self.image.name
        self.tab_frame = tk.Frame(self.tab_control)
        self.tab_control.add(self.tab_frame, text=label_name, padding=10)
        self.tab_control.pack(expand=1, fill="both")
        cimage: CImage = self.image

        if cimage.type in [CImage.IMAGE_TYPE.RGB_INT, CImage.IMAGE_TYPE.RGB_DOUBLE]:
            data = cimage.data
        else:
            data = cimage.ycbcr2rgb().data.astype(np.uint8)

        img = Image.fromarray(data)
        photo_img = ImageTk.PhotoImage(image=img)

        img_label = tk.Label(self.tab_frame, image=photo_img)
        img_label.image = photo_img
        img_label.pack()
        return self

    def setup_kernel_tab(self):
        """Setup for kernel tab."""
        label_name: str = self.image.name
        self.tab_frame = tk.Frame(self.tab_control)
        self.tab_control.add(self.tab_frame, text=label_name, padding=10)
        self.tab_control.pack(expand=1, fill="both")

        img = Image.fromarray(self.kernel)
        photo_img = ImageTk.PhotoImage(image=img)

        img_label = tk.Label(self.tab_frame, image=photo_img)
        img_label.image = photo_img
        img_label.pack()
        return self
