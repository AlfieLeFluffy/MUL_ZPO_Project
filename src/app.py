import os 
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog

from src.processing import ImageProcessor
from src.image import CImage
from src.gui.tab import Tab
from src.gui.status_bar import StatusBar

class GUI_App:

    MAIN_TAB: str = "Main"

    def __init__(self, _tkroot):
        self.current_tab = None
        self.tabs = {}

        self.tkroot = _tkroot

        self.setup_gui_basic()
        self.setup_gui_elements()
        
    def setup_gui_basic(self):
        self.tkroot.title("Fourier Tranform Filtering")
        self.tkroot.geometry('800x600')

    def setup_gui_elements(self):
        self.setup_gui_menu()

        self.tab_control: ttk.Notebook = ttk.Notebook(self.tkroot)
        self.tab_control.bind("<<NotebookTabChanged>>", self.on_tab_change)

        self.setup_initial_tab()
        self.statusbar: StatusBar = StatusBar()
    
    def setup_initial_tab(self):
        if not self.MAIN_TAB in self.tabs:
            self.tabs[self.MAIN_TAB] = Tab(self.tab_control, _menu_tab = True, _menu_controls = {"Open File": self.open_file})

    def on_tab_change(self, event):
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
    
    def add_tab(self, _tab_name):
        new_tab = tk.Frame(self.tab_control)
        self.tabs[_tab_name] = new_tab
        self.tab_control.add(new_tab, text=_tab_name, padding=10)
        self.tab_control.pack(expand = 1, fill ="both")
        return new_tab

    def remove_tab(self, _tab_name):
        if _tab_name in self.tabs:
            self.tab_control.forget(self.tabs[_tab_name].tab_frame)
            if _tab_name in self.tabs:
                del self.tabs[_tab_name]
            if _tab_name in self.tabs:
                del self.tabs[_tab_name]

    def close_current_tab(self):
        if self.tabs:
            tab_name = self.tab_control.tab(self.tab_control.select(), "text")
            if self.tabs[tab_name].saved:
                self.remove_tab(tab_name)
            else:
                user_input = messagebox.askyesno(title="Close unsaved file", message="Do you want to close an unsaved file?")
                if not user_input:
                    return
                else:
                    self.remove_tab(tab_name)

    def setup_gui_menu(self):
        self.main_menu = tk.Menu(self.tkroot)
        self.fileMenu = tk.Menu(self.main_menu, tearoff=0)
        self.editMenu = tk.Menu(self.main_menu, tearoff=0)
        self.tabMenu = tk.Menu(self.main_menu, tearoff=0)
        self.tkroot.config(menu=self.main_menu)

        self.main_menu.add_cascade(label="File", menu=self.fileMenu)
        self.main_menu.add_cascade(label="Edit", menu=self.editMenu)
        self.main_menu.add_cascade(label="Tab", menu=self.tabMenu)

        fileOptions = [("Open File", self.open_file), ("Save Image", self.save_file), "separator", ("About", self.show_popup_about), ("Exit", self.tkroot.destroy)]
        editOptions = [("Update Image", self.update_image), ("Deconvolve Image", self.deconvolve_image)]
        tabOptions = [("Open Menu", self.setup_initial_tab),("Close Tab", self.close_current_tab)]
        
        menuOptions = [(self.fileMenu, fileOptions), (self.editMenu, editOptions), (self.tabMenu, tabOptions)]

        for menu in menuOptions:
            if not isinstance(menu, tuple) or len(menu) != 2:
                raise Exception("Invalid menu option. Expected a tuple of (menu, options).")
            self.setup_gui_menu_options(menu)
    
    def setup_gui_menu_options(self, _menu):
        for option in _menu[1]:
            if isinstance(option, tuple):
                if len(option) != 2 or not callable(option[1]):
                    raise Exception("Invalid menu option tuple. Expected (label, command) where command is a callable.")
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
        file_path = filedialog.askopenfilename()
        if not file_path or not os.path.isfile(file_path):
            return
        
        img = CImage.load_image(file_path)
        tab = Tab(self.tab_control, img, True)
        self.tabs[img.name] = tab
        self.current_tab = img.name

        self.select_tab(img.name)
    
    def save_file(self):
        assert self.current_tab in self.tabs, f"save_image: No image of name {self.current_tab} in self.images."

        if self.current_tab not in self.tabs:
            self.trow_error("No processed image available. Please process an image first.")
            return

        if self.tabs[self.current_tab].menu_tab:
            return

        output_path = filedialog.asksaveasfilename(defaultextension=".jpeg")
        if output_path:
            self.tabs[self.current_tab].image.save_image(output_path)
            self.tabs[self.current_tab].saved = True

    def update_image(self):
        if self.current_tab not in self.tabs:
            self.trow_error("No image loaded. Please open an image first.")
            return
        
        kernel = cv2.getGaussianKernel(9, 3) @ cv2.getGaussianKernel(9, 3).T
        kernel = kernel / np.sum(kernel)

        orig_image: CImage = self.tabs[self.current_tab].image
        proc_image = ImageProcessor.apply_kernel_to_image(orig_image, kernel)
        self.tabs[proc_image.name] = Tab(self.tab_control, proc_image)

        self.select_tab(proc_image.name)
    
    def select_tab(self, tab_name):
        if tab_name in self.tabs:
            self.tab_control.select(self.tabs[tab_name].tab_frame)
    
    def deconvolve_image(self):
        if self.current_tab not in self.tabs:
            self.trow_error("No image loaded. Please open an image first.")
            return

        orig_image: CImage = self.tabs[self.current_tab].image
        kernel_size = simpledialog.askstring(title="Kernel Size", prompt="What should be the kernel size?:")
        verbose = messagebox.askyesno(title="Verbose Setting", message="Should the algorithm be verbose?")
        params = ImageProcessor.Deconvolve.MinRankKernel.MinRankKernelParam()
        params.verbose = verbose
        decon_image, kernel = ImageProcessor.deconvolve_image(orig_image, int(kernel_size), params)
        self.tabs[decon_image.name] = Tab(self.tab_control, decon_image)

        self.select_tab(decon_image.name)

    def trow_error(self, _message):
        messagebox.showerror("Error", _message)

    def show_popup_about(self):
        about_window = tk.Toplevel(self.tkroot)
        about_window.title("About ZPO/MUL Project")
        tk.Label(about_window, text="This is a project that aims to demostrate the use of FFT/IFFT in image processing.").pack(padx=20, pady=20)

def main():
    tkroot = tk.Tk()
    app = GUI_App(tkroot)
    tkroot.mainloop()

if __name__ == "__main__":
    main()