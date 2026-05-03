import os
import time
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
from src.util.parallel_exec import ctk_par_start, ctk_par_stop, async_execute
from src.conv.spectral_convolution import SpectralConvolution
from src.conv.kernel_generation import KernelGeneration


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
                _menu_controls={
                    "Open File": self.open_file,
                    "Run Test": self.run_tests,
                    "Exit": self.tkroot.destroy,
                },
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

        cascade_menus = ["File", "Edit", "Filter", "Tab"]

        self.menus = {}
        for cascade in cascade_menus:
            self.menus[cascade] = self.menu_bar.add_cascade(cascade)

        fileOptions = [
            ("Open File", self.open_file),
            ("Save Image", self.save_file),
            "separator",
            ("Run Tests", self.run_tests),
            ("About", self.show_popup_about),
            ("Exit", self.tkroot.destroy),
        ]
        editOptions = [
            ("Grayscale Image", self.grayscale_image),
            ("Convolve Image", self.convolve_image),
            ("Deconvolve Image", self.deconvolve_image),
        ]
        filterOptions = [
            ("High-Pass", self.filter_high_pass),
            ("Low-Pass", self.filter_low_pass),
            ("Hamming-Pass", self.filter_hamming),
        ]
        tabOptions = [
            ("Open Menu", self.setup_initial_tab),
            ("Close Tab", self.close_current_tab),
        ]

        menuOptions = [
            fileOptions,
            editOptions,
            filterOptions,
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

        if self.tabs[self.tab_control.get()].type == Tab.TabType.MENU:
            return

        output_path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if output_path:
            self.tabs[self.tab_control.get()].image.save_image(output_path)
            self.tabs[self.tab_control.get()].saved = True

    def grayscale_image(self):
        """Takes the current image and updates it with a kernel."""

        orig_image: CImage = self.get_current_cimage()
        if orig_image is None:
            return

        async_execute(self.grayscale_image_async(orig_image))

    async def grayscale_image_async(self, _orig_image: CImage):
        """Grayscale async task.

        Args:
            _orig_image (CImage): Input image
            _kernel (np.ndarray): Input kernel
        """
        proc_image = ImageProcessor.grayscale_cimage(_orig_image)
        self.tabs[proc_image.name] = Tab(self.tab_control, proc_image)
        self.select_tab(proc_image.name)

    def convolve_image(self):
        """Takes the current image and updates it with a kernel."""

        (kernel_type, kernel_size, kernel_sigma) = Form(
            self.tkroot, Form.FormType.CONV
        ).popup()

        kernel = KernelGeneration.create_kernel(
            kernel_type,
            (kernel_size, kernel_size),
            kernel_sigma,
        )

        orig_image: CImage = self.get_current_cimage()
        if orig_image is None:
            return

        async_execute(self.convolve_image_async(orig_image, kernel))

    async def convolve_image_async(self, _orig_image: CImage, _kernel: np.ndarray):
        """Update async task.

        Args:
            _orig_image (CImage): Input image
            _kernel (np.ndarray): Input kernel
        """
        proc_image = ImageProcessor.convolve(_orig_image, _kernel)
        self.tabs[proc_image.name] = Tab(self.tab_control, proc_image)
        self.select_tab(proc_image.name)

    def deconvolve_image(self):
        """Deconvolves the current image using the MRK and Bregmen deconvolution with given parameters"""

        orig_image: CImage = self.get_current_cimage()
        if orig_image is None:
            return

        (kernel_size, params) = Form(self.tkroot, Form.FormType.MRK_DECONV).popup()

        if kernel_size is None or params is None:
            return

        if kernel_size <= 3 or kernel_size % 2 == 0:
            self.throw_error("Kernel size must be an odd integer greater than 3.")
            return

        async_execute(self.run_deconvolution_async(orig_image, kernel_size, params))

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
        self.tabs[decon_image.name] = Tab(self.tab_control, decon_image)
        self.select_tab(decon_image.name)

    def filter_high_pass(self):
        """Filter current image with a high pass filter"""
        orig_image: CImage = self.get_current_cimage()
        if orig_image is None:
            return

        offset_size = Form(self.tkroot, Form.FormType.FILTER).popup()

        if not offset_size:
            return

        filter_image_data = SpectralConvolution.fft_filter(
            orig_image.data,
            KernelGeneration.FilterType.HIGH_PASS,
            int(offset_size),
        )
        filter_image = CImage(
            filter_image_data.astype(np.uint8), orig_image.name, _stamp=True
        )
        self.tabs[filter_image.name] = Tab(self.tab_control, filter_image)
        self.select_tab(filter_image.name)

    def filter_low_pass(self):
        """Filter current image with a low pass filter"""
        orig_image: CImage = self.get_current_cimage()
        if orig_image is None:
            return

        offset_size = Form(self.tkroot, Form.FormType.FILTER).popup()

        if not offset_size:
            return

        filter_image_data = SpectralConvolution.fft_filter(
            orig_image.data,
            KernelGeneration.FilterType.LOW_PASS,
            int(offset_size),
        )
        filter_image = CImage(
            filter_image_data.astype(np.uint8), orig_image.name, _stamp=True
        )
        self.tabs[filter_image.name] = Tab(self.tab_control, filter_image)
        self.select_tab(filter_image.name)

    def filter_hamming(self):
        """Filter current image with hamming window"""
        orig_image: CImage = self.get_current_cimage()
        if orig_image is None:
            return

        offset_size = Form(self.tkroot, Form.FormType.FILTER).popup()

        if not offset_size:
            return

        filter_image_data = SpectralConvolution.fft_filter(
            orig_image.data,
            KernelGeneration.FilterType.HAMMING,
            int(offset_size),
        )
        filter_image = CImage(
            filter_image_data.astype(np.uint8), orig_image.name, _stamp=True
        )
        self.tabs[filter_image.name] = Tab(self.tab_control, filter_image)
        self.select_tab(filter_image.name)

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

    def run_tests(self):
        """Tests the implementation of fft_convolve against other similar methods"""
        async_execute(self.run_tests_async())

    async def run_tests_async(self):
        """Runs test task async"""
        if not os.path.exists("tests"):
            os.makedirs("tests")

        test_image = self.get_current_cimage()
        if test_image is None:
            test_image = CImage(
                np.random.randint(0, 255, (1024, 1024, 1), dtype=np.uint8), "test_image"
            )

        number_of_cycles, kernel_size, kernel_sigma = Form(
            self.tkroot, Form.FormType.TESTING
        ).popup()

        kernel = KernelGeneration.create_kernel(
            KernelGeneration.KernelType.GAUSS_BLUR,
            (kernel_size, kernel_size),
            kernel_sigma,
        )

        from scipy import signal

        methods = [
            (
                "mine fft_convolve same",
                lambda img, ker: SpectralConvolution.fft_convolve(
                    img, ker, mode="same"
                ),
            ),
            (
                "mine fft_convolve full",
                lambda img, ker: SpectralConvolution.fft_convolve(
                    img, ker, mode="full"
                ),
            ),
            (
                "mine fft_convolve valid",
                lambda img, ker: SpectralConvolution.fft_convolve(
                    img, ker, mode="valid"
                ),
            ),
            (
                "scipy convolve2d same",
                lambda img, ker: signal.convolve2d(img, ker, "same"),
            ),
            (
                "scipy convolve2d full",
                lambda img, ker: signal.convolve2d(img, ker, "full"),
            ),
            (
                "scipy convolve2d valid",
                lambda img, ker: signal.convolve2d(img, ker, "valid"),
            ),
            (
                "scipy fftconvolve same",
                lambda img, ker: signal.fftconvolve(img, ker, "same"),
            ),
            (
                "scipy fftconvolve full",
                lambda img, ker: signal.fftconvolve(img, ker, "full"),
            ),
            (
                "scipy fftconvolve valid",
                lambda img, ker: signal.fftconvolve(img, ker, "valid"),
            ),
        ]

        log_file = os.path.join(
            "tests", time.strftime("%Y_%m_%d-%H_%M_%S-") + "convolution_performance.log"
        )
        cycles = number_of_cycles
        with open(log_file, "w") as f:
            f.write("Convolution Performance Test Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Image size: {test_image.data.shape}\n")
            f.write(f"Kernel size: {kernel.shape}\n")
            f.write(f"Test run on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Cycles per test: {cycles}\n\n")

            for name, func in methods:
                try:
                    print(f"Running test {name}")
                    start_time = time.time()
                    for _ in range(cycles):
                        result = func(test_image.data[:, :, 0].astype(float), kernel)
                    end_time = time.time()
                    elapsed = end_time - start_time
                    f.write(f"{name}: {elapsed:.4f} seconds\n")
                    f.write(f"  Output shape: {result.shape}\n")
                except Exception as e:
                    f.write(f"{name}: Error - {str(e)}\n")
                f.write("\n")


def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    tkroot = ctk.CTk()
    async_loop = asyncio.get_event_loop()
    GUI_App(tkroot, async_loop)
    ctk_par_start()
    tkroot.mainloop()
    ctk_par_stop()


if __name__ == "__main__":
    main()
