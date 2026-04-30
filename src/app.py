import os
import cv2
import numpy as np
import tkinter as tk
import platform as pf
import customtkinter as ctk
import CTkMenuBar as ctkmb
import asyncio
from tkinter import filedialog, messagebox

from src.processing import ImageProcessor
from src.image import CImage
from src.gui.tab import Tab
from src.gui.status_bar import StatusBar
from src.gui.form import Form
from src.util.parallel import Parallel


class GUI_App:
    MAIN_TAB: str = "Main"

    def __init__(self, _tkroot, _async_loop):
        self.tabs = {}

        self.tkroot = _tkroot
        self.async_loop = _async_loop

        self.setup_gui_basic()
        self.setup_gui_elements()

    def setup_gui_basic(self):
        """Setup for basic GUI application settings."""
        self.tkroot.title("FTF")
        self.tkroot.geometry("800x600")
        self.tkroot.minsize(600, 400)

    def setup_gui_elements(self):
        """Setup for basic GUI elements."""
        self.setup_gui_menu()

        self.tab_control: ctk.CTkTabview = ctk.CTkTabview(
            self.tkroot, anchor="nw", command=self._on_tab_change
        )
        self.tab_control.pack(expand=True, anchor="nw", fill="both")

        self.setup_initial_tab()
        self.statusbar: StatusBar = StatusBar(self.tkroot)

    def _on_tab_change(self):
        """Function to be called when the tab is changed. Updates the status bar."""
        if self.tab_control.get() == self.MAIN_TAB:
            self.statusbar.reset_status_bar()
        else:
            self.statusbar.update_status_bar(self.get_current_cimage())

    def setup_initial_tab(self):
        """Setup initial menu tab"""
        if self.MAIN_TAB not in self.tabs:
            self.tabs[self.MAIN_TAB] = Tab(
                self.tab_control,
                _menu_tab=True,
                _menu_controls={"Open File": self.open_file},
            )

    def remove_tab(self, _tab_name: str):
        """Removes a tab from the tab control and other storage based on its name.

        Args:
            _tab_name (str): Name of the target tab
        """
        if _tab_name in self.tabs:
            self.tab_control.delete(_tab_name)
            if _tab_name in self.tabs:
                del self.tabs[_tab_name]
            if _tab_name in self.tabs:
                del self.tabs[_tab_name]
            self._on_tab_change()

    def close_current_tab(self):
        """Closes the current tab. This function can check if the current image file has been saved and warns the user."""
        if not self.tabs:
            return
        tab_name = self.tab_control.get()
        if self.tabs[tab_name].is_saved():
            self.remove_tab(tab_name)
        else:
            user_input = messagebox.askyesno(
                title="Close unsaved file",
                message="Do you want to close an unsaved file?",
            )
            if not user_input:
                return
            else:
                self.remove_tab(tab_name)

    def check_current_tab(self):
        """Checks if the current tab is valid and exists"""
        if self.tab_control.get() not in self.tabs:
            return False
        tab = self.tabs[self.tab_control.get()]
        if not isinstance(tab, Tab):
            print(tab)
            raise Exception("check_current_tab: Something other then a Tab found.")
        return True

    def check_current_tab_image(self):
        """Checks if the current tab has an image"""
        if not self.check_current_tab():
            return False
        tab: Tab = self.tabs[self.tab_control.get()]
        if tab.type == Tab.TabType.MENU:
            return False
        if not tab.image:
            return False
        return True

    def get_current_cimage(self):
        """Gets the current image from the current tab"""
        if not self.check_current_tab_image():
            return None
        tab: Tab = self.tabs[self.tab_control.get()]
        return tab.image

    def setup_gui_menu(self):
        """Setup for GUI menu.

        Raises:
            Exception: Invalid menu options
        """
        match pf.system():
            case "Linux":
                self.menu_bar = ctkmb.CTkMenuBar(self.tkroot)
            case "Windows":
                self.menu_bar = ctkmb.CTkTitleMenu(self.tkroot)
            case _:
                self.menu_bar = ctkmb.CTkMenuBar(self.tkroot)

        self.tkroot.config(menu=self.menu_bar)

        cascade_menus = ["File", "Edit", "Tab"]

        self.menus = {}
        for cascade in cascade_menus:
            self.menus[cascade] = self.menu_bar.add_cascade(cascade)

        fileOptions = [
            ("Open File", self.open_file),
            ("Save Image", self.save_file),
            "separator",
            ("About", self.show_popup_about),
            ("Exit", self.tkroot.destroy),
        ]
        editOptions = [
            ("Update Image", self.update_image),
            ("Deconvolve Image", self.deconvolve_image),
        ]
        tabOptions = [
            ("Open Menu", self.setup_initial_tab),
            ("Close Tab", self.close_current_tab),
        ]

        menuOptions = [
            fileOptions,
            editOptions,
            tabOptions,
        ]

        for cascade in zip(cascade_menus, menuOptions):
            self.setup_gui_menu_options(cascade)

    def setup_gui_menu_options(self, _cascade):
        """Creates a menu list from a list of tuples containing button names and functions.

        Args:
            _cascade (tuple): A tuple containing the cascade name and its options

        Raises:
            Exception: Invalid tuple
            Exception: Invalid option type
        """
        dropdown = ctkmb.CustomDropdownMenu(widget=self.menus[_cascade[0]])
        for option in _cascade[1]:
            if option == "separator":
                dropdown.add_separator()
            elif isinstance(option, tuple) and len(option) == 2:
                dropdown.add_option(option=option[0], command=option[1])
            else:
                raise Exception("Invalid menu option: " + str(option))

    def open_file(self):
        """Through a file selection dialog opens an image file and saves it."""
        file_path = filedialog.askopenfilename()
        if not file_path or not os.path.isfile(file_path):
            return

        img = CImage.load_image(file_path)
        tab = Tab(self.tab_control, img, True)
        self.tabs[img.name] = tab

        self.select_tab(img.name)

    def save_file(self):
        """Saves a current image file through."""
        assert self.tab_control.get() in self.tabs, (
            f"save_image: No image of name {self.tab_control.get()} in self.images."
        )

        if self.tab_control.get() not in self.tabs:
            self.throw_error(
                "No processed image available. Please process an image first."
            )
            return

        if self.tabs[self.tab_control.get()].menu_tab:
            return

        output_path = filedialog.asksaveasfilename(defaultextension=".jpeg")
        if output_path:
            self.tabs[self.tab_control.get()].image.save_image(output_path)
            self.tabs[self.tab_control.get()].saved = True

    def update_image(self):
        """Takes the current image and updates it with a kernel."""
        kernel = cv2.getGaussianKernel(9, 3) @ cv2.getGaussianKernel(9, 3).T
        kernel = kernel / np.sum(kernel)

        orig_image: CImage = self.get_current_cimage()
        if orig_image is None:
            return

        proc_image = ImageProcessor.filter_cimage(orig_image, kernel)
        self.tabs[proc_image.name] = Tab(self.tab_control, proc_image)

        self.select_tab(proc_image.name)

    def select_tab(self, _tab_name: str):
        """Changes the currently selected tab within the tab controls.

        Args:
            tab_name (str): Name of the target tab
        """
        if _tab_name in self.tabs:
            self.tab_control.set(_tab_name)
        if _tab_name == self.MAIN_TAB:
            self.statusbar.reset_status_bar()
        else:
            self.statusbar.update_status_bar(self.get_current_cimage())

    def deconvolve_image(self):
        """Deconvolves the current image using the MRK and Bregmen deconvolution with given parameters"""

        orig_image: CImage = self.get_current_cimage()
        if orig_image is None:
            return

        (kernel_size, params) = Form(self.tkroot, Form.FormType.DECONV).popup()

        if kernel_size is None or params is None:
            return

        Parallel._run_async_task(
            self.async_loop,
            self.run_deconvolution_async(orig_image, kernel_size, params),
        )

    async def run_deconvolution_async(self, _image: CImage, _kernel_size: int, _params):
        """Deconvolution async task.

        Args:
            _image (CImage): Input image
            _kernel_size (int): Expected kernel size
            _params (MinRankKernelParam): Algorithm parameters
        """
        decon_image, kernel = ImageProcessor.deconvolve_cimage(
            _image,
            int(_kernel_size),
            _params,
        )
        self.tabs[decon_image.name] = Tab(self.tab_control, decon_image, _kernel=kernel)
        self.select_tab(decon_image.name)

    def throw_error(self, _message: str):
        """Throws error message popup

        Args:
            _message (str): Error message
        """
        messagebox.showerror("Error", _message)

    def show_popup_about(self):
        """Shows project about window"""
        about_window = tk.Toplevel(self.tkroot)
        about_window.title("About ZPO/MUL Project")
        tk.Label(
            about_window,
            text="This is a project that aims to demostrate the use of FFT/IFFT in image processing.",
        ).pack(padx=20, pady=20)


def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    tkroot = ctk.CTk()
    async_loop = asyncio.get_event_loop()
    GUI_App(tkroot, async_loop)
    tkroot.mainloop()


if __name__ == "__main__":
    main()
