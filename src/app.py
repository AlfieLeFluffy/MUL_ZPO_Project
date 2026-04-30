import os
import cv2
import numpy as np
import tkinter as tk
import asyncio
import threading
from tkinter import filedialog, messagebox, ttk, simpledialog

from src.processing import ImageProcessor
from src.image import CImage
from src.gui.tab import Tab
from src.gui.status_bar import StatusBar


class GUI_App:
    MAIN_TAB: str = "Main"

    def __init__(self, _tkroot, _async_loop):
        self.current_tab = None
        self.tabs = {}

        self.tkroot = _tkroot
        self.async_loop = _async_loop

        self.setup_gui_basic()
        self.setup_gui_elements()

    def setup_gui_basic(self):
        """Setup for basic GUI application settings."""
        self.tkroot.title("Fourier Tranform Filtering")
        self.tkroot.geometry("800x600")

    def setup_gui_elements(self):
        """Setup for basic GUI elements."""
        self.setup_gui_menu()

        self.tab_control: ttk.Notebook = ttk.Notebook(self.tkroot)
        self.tab_control.bind("<<NotebookTabChanged>>", self.on_tab_change)

        self.setup_initial_tab()
        self.statusbar: StatusBar = StatusBar()

    def setup_initial_tab(self):
        """Setup initial menu tab"""
        if self.MAIN_TAB not in self.tabs:
            self.tabs[self.MAIN_TAB] = Tab(
                self.tab_control,
                _menu_tab=True,
                _menu_controls={"Open File": self.open_file},
            )

    def on_tab_change(self, event):
        """Runs every time a tab changes.

        Args:
            event (_type_): Incoming event
        """
        if len(self.tab_control.tabs()) == 0:
            tab_text = None
        else:
            tab_text = self.tab_control.tab("current", "text")
        self.current_tab = tab_text
        if self.current_tab == self.MAIN_TAB:
            self.statusbar.reset_status_bar()
            return
        if self.current_tab in self.tabs:
            self.statusbar.update_status_bar(self.tabs[self.current_tab].image)

    def add_tab(self, _tab_name: str):
        """Add a tab to tab control with the name _tab_name.

        Args:
            _tab_name (str): The name of the tab to add.

        Returns:
            tk.Frame: The newly created tab frame
        """
        if _tab_name in self.tabs:
            raise Exception(f"Tab with name {_tab_name} already exists.")
        new_tab = tk.Frame(self.tab_control)
        self.tabs[_tab_name] = new_tab
        self.tab_control.add(new_tab, text=_tab_name, padding=10)
        self.tab_control.pack(expand=1, fill="both")
        return new_tab

    def remove_tab(self, _tab_name: str):
        """Removes a tab from the tab control and other storage based on its name.

        Args:
            _tab_name (str): Name of the target tab
        """
        if _tab_name in self.tabs:
            self.tab_control.forget(self.tabs[_tab_name].tab_frame)
            if _tab_name in self.tabs:
                del self.tabs[_tab_name]
            if _tab_name in self.tabs:
                del self.tabs[_tab_name]

    def close_current_tab(self):
        """Closes the current tab. This function can check if the current image file has been saved and warns the user."""
        if not self.tabs:
            return
        tab_name = self.tab_control.tab(self.tab_control.select(), "text")
        if self.tabs[tab_name].saved:
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
        if self.current_tab not in self.tabs:
            return False
        tab = self.tabs[self.current_tab]
        if not isinstance(tab, Tab):
            print(tab)
            raise Exception("check_current_tab: Something other then a Tab found.")
        return True

    def check_current_tab_image(self):
        if not self.check_current_tab():
            return False
        tab: Tab = self.tabs[self.current_tab]
        if tab.type == Tab.TabType.MENU:
            return False
        if not tab.image:
            return False
        return True

    def setup_gui_menu(self):
        """Setup for GUI menu.

        Raises:
            Exception: Invalid menu options
        """
        self.main_menu = tk.Menu(self.tkroot)
        self.fileMenu = tk.Menu(self.main_menu, tearoff=0)
        self.editMenu = tk.Menu(self.main_menu, tearoff=0)
        self.tabMenu = tk.Menu(self.main_menu, tearoff=0)
        self.tkroot.config(menu=self.main_menu)

        self.main_menu.add_cascade(label="File", menu=self.fileMenu)
        self.main_menu.add_cascade(label="Edit", menu=self.editMenu)
        self.main_menu.add_cascade(label="Tab", menu=self.tabMenu)

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
            (self.fileMenu, fileOptions),
            (self.editMenu, editOptions),
            (self.tabMenu, tabOptions),
        ]

        for menu in menuOptions:
            if not isinstance(menu, tuple) or len(menu) != 2:
                raise Exception(
                    "Invalid menu option. Expected a tuple of (menu, options)."
                )
            self.setup_gui_menu_options(menu)

    def setup_gui_menu_options(self, _menu):
        """Creates a menu list from a list of tuples containing button names and functions.

        Args:
            _menu ([(str, fn)]): An array of string names and functions

        Raises:
            Exception: Invalid tuple
            Exception: Invalud option type
        """
        for option in _menu[1]:
            if isinstance(option, tuple):
                if len(option) != 2 or not callable(option[1]):
                    raise Exception(
                        "Invalid menu option tuple. Expected (label, command) where command is a callable."
                    )
                _menu[0].add_command(label=option[0], command=option[1])
            elif isinstance(option, str):
                match option:
                    case "separator":
                        _menu[0].add_separator()
                    case _:
                        _menu[0].add_command(label=option)
            else:
                raise Exception("Invalid menu option type. Expected tuple or string.")

    def open_file(self):
        """Through a file selection dialog opens an image file and saves it."""
        file_path = filedialog.askopenfilename()
        if not file_path or not os.path.isfile(file_path):
            return

        img = CImage.load_image(file_path)
        tab = Tab(self.tab_control, img, True)
        self.tabs[img.name] = tab
        self.current_tab = img.name

        self.select_tab(img.name)

    def save_file(self):
        """Saves a current image file through."""
        assert self.current_tab in self.tabs, (
            f"save_image: No image of name {self.current_tab} in self.images."
        )

        if self.current_tab not in self.tabs:
            self.trow_error(
                "No processed image available. Please process an image first."
            )
            return

        if self.tabs[self.current_tab].menu_tab:
            return

        output_path = filedialog.asksaveasfilename(defaultextension=".jpeg")
        if output_path:
            self.tabs[self.current_tab].image.save_image(output_path)
            self.tabs[self.current_tab].saved = True

    def update_image(self):
        if not self.check_current_tab_image():
            return

        kernel = cv2.getGaussianKernel(9, 3) @ cv2.getGaussianKernel(9, 3).T
        kernel = kernel / np.sum(kernel)

        orig_image: CImage = self.tabs[self.current_tab].image
        proc_image = ImageProcessor.filter_cimage(orig_image, kernel)
        self.tabs[proc_image.name] = Tab(self.tab_control, proc_image)

        self.select_tab(proc_image.name)

    def select_tab(self, tab_name: str):
        """Changes the currently selected tab within the tab controls.

        Args:
            tab_name (str): Name of the target tab
        """
        if tab_name in self.tabs:
            self.tab_control.select(self.tabs[tab_name].tab_frame)

    def _run_async_task(self, fun):
        threading.Thread(target=lambda: self.async_loop.run_until_complete(fun)).start()

    def deconvolve_image(self):
        if not self.check_current_tab_image():
            return

        orig_image: CImage = self.tabs[self.current_tab].image
        kernel_size = simpledialog.askstring(
            title="Kernel Size", prompt="What should be the kernel size?:"
        )
        verbose = messagebox.askyesno(
            title="Verbose Setting", message="Should the algorithm be verbose?"
        )
        kernel_size = kernel_size if kernel_size else 9
        params = ImageProcessor.Deconvolve.MinRankKernel.MinRankKernelParam()
        params.verbose = verbose
        self._run_async_task(
            self.run_deconvolution_async(orig_image, kernel_size, params)
        )

    async def run_deconvolution_async(self, _image, _kernel_size, _params):
        decon_image, kernel = ImageProcessor.deconvolve_cimage(
            _image,
            int(_kernel_size),
            _params,
        )
        self.tabs[decon_image.name] = Tab(self.tab_control, decon_image, _kernel=kernel)
        self.select_tab(decon_image.name)

    def trow_error(self, _message):
        messagebox.showerror("Error", _message)

    def show_popup_about(self):
        about_window = tk.Toplevel(self.tkroot)
        about_window.title("About ZPO/MUL Project")
        tk.Label(
            about_window,
            text="This is a project that aims to demostrate the use of FFT/IFFT in image processing.",
        ).pack(padx=20, pady=20)


def main():
    tkroot = tk.Tk()
    async_loop = asyncio.get_event_loop()
    GUI_App(tkroot, async_loop)
    tkroot.mainloop()


if __name__ == "__main__":
    main()
