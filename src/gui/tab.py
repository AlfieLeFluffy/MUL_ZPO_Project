import customtkinter as ctk
import numpy as np
from enum import Enum
from PIL import Image

from src.image import CImage


class Tab:
    MAIN_TAB: str = "Main"

    class TabType(Enum):
        MENU = 1
        IMAGE = 2
        KERNEL = 3

    def __init__(
        self,
        _tab_control: ctk.CTkTabview,
        _image: CImage = None,
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
        self.tab_control: ctk.CTkTabview = _tab_control
        self.type = self.TabType.MENU if _menu_tab else self.TabType.IMAGE
        self.menu_controls = _menu_controls
        self.image: CImage = _image
        self.kernel: np.ndarray = _kernel

        if self.type == self.TabType.MENU and self.menu_controls is None:
            raise Exception("Menu tab must have menu controls.")

        if (
            self.type == self.TabType.IMAGE
            and self.image is None
            and self.kernel is None
        ):
            self.type = self.TabType.KERNEL

        if self.type == self.TabType.IMAGE and self.image is None:
            raise Exception("Image tab must have an image.")

        if self.type == self.TabType.MENU and self.image is not None:
            raise Exception("Menu tab cannot have an image.")

        if self.type == self.TabType.IMAGE and self.menu_controls is not None:
            raise Exception("Image tab cannot have menu controls.")

        match self.type:
            case self.TabType.MENU:
                self.setup_menu_tab()
            case self.TabType.IMAGE:
                self.setup_image_tab()
            case self.TabType.KERNEL:
                self.setup_kernel_tab()

    def is_saved(self):
        """Returns whether the image in the tab is saved."""
        if self.type == self.TabType.MENU:
            return True
        elif self.type in [self.TabType.IMAGE]:
            return self.image.is_saved()
        else:
            return True

    def setup_menu_tab(self):
        """Setup for a menu tab"""
        self.saved = True
        self.tab_control.add(self.MAIN_TAB)
        self.tab_frame = ctk.CTkFrame(self.tab_control.tab(self.MAIN_TAB))
        self.menu_label = ctk.CTkLabel(
            self.tab_frame,
            text="Image Deconvolution and Fourier Transform \nin Image Filtering",
            font=("test", 20),
        ).pack(anchor="c", padx=10, pady=10)
        for key in self.menu_controls.keys():
            button = ctk.CTkButton(
                self.tab_frame, text=key, command=self.menu_controls[key]
            )
            button.pack(anchor="c", padx=4, pady=4)
        self.tab_frame.pack(padx=10, pady=10, fill="both", expand=True)
        return self

    def setup_image_tab(self):
        """Setup for image tab."""
        self.create_tab_frame()
        self.create_image_label(self.tab_frame, self.image)
        self.tab_frame.pack(padx=10, pady=10, fill="both", expand=True)
        return self

    def setup_kernel_tab(self):
        """Setup for kernel tab."""
        self.create_tab_frame()
        image = Image.fromarray(self.kernel)
        self.ctkimage = ctk.CTkImage(
            light_image=image, dark_image=image, size=image.size
        )
        self.image_label = ctk.CTkLabel(self.tab_frame, text="", image=self.ctkimage)
        self.image_label.pack(anchor="center", fill="both")
        self.tab_frame.pack(padx=10, pady=10, fill="both", expand=True)
        return self

    def create_tab_frame(self):
        """Creates a new tab frame."""
        self.tab_control.add(self.image.name)
        self.tab_frame = ctk.CTkFrame(self.tab_control.tab(self.image.name))

    def create_image_label(self, _frame: ctk.CTkFrame, _image: CImage):
        """Creates an image label for the given image.

        Args:
            _frame (ctk.CTkFrame): Frame to create the label in
            _image (CImage): Image to create the label for
        """
        cimage: CImage = _image

        if cimage.type in [CImage.IMAGE_TYPE.RGB_INT, CImage.IMAGE_TYPE.RGB_DOUBLE]:
            data = cimage.data
        else:
            data = cimage.ycbcr2rgb().data.astype(np.uint8)

        image = Image.fromarray(data)
        self.ctkimage = ctk.CTkImage(
            light_image=image, dark_image=image, size=image.size
        )
        self.image_label = ctk.CTkLabel(_frame, text="", image=self.ctkimage)
        self.image_label.pack(anchor="center", fill="both")

    def update_image_label(self, _frame: ctk.CTkFrame, _image: CImage):
        """Updates the image in the tab.

        Args:
            _frame (ctk.CTkFrame): Frame containing the image label
            _image (CImage): New image to be displayed
        """
        self.image = _image
        cimage: CImage = self.image

        if cimage.type in [CImage.IMAGE_TYPE.RGB_INT, CImage.IMAGE_TYPE.RGB_DOUBLE]:
            data = cimage.data
        else:
            data = cimage.ycbcr2rgb().data.astype(np.uint8)

        image = Image.fromarray(data)
        self.ctkimage = ctk.CTkImage(
            light_image=image, dark_image=image, size=image.size
        )
        self.image_label.configure(image=self.ctkimage)
