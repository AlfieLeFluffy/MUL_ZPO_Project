import os 
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk

from src.processing import ImageProcessor
from src.image import CImage

class GUI_App:
    def __init__(self, _tkroot):
        self.current_tab = None
        self.tabs = {}
        self.images = {}

        self.tkroot = _tkroot

        self.setup_gui_basic()
        self.setup_gui_elements()
        
    def setup_gui_basic(self):
        self.tkroot.title("Fourier Tranform Filtering")
        self.tkroot.geometry('800x600')

    def on_tab_change(self, event):
        if len(self.tabControl.tabs()) == 0:
            tab_text = None
        else:
            tab_text = self.tabControl.tab("current", "text")
        self.current_tab = tab_text

    def setup_gui_elements(self):
        self.setup_gui_menu()

        self.tabControl: ttk.Notebook = ttk.Notebook(self.tkroot)
        self.tabControl.bind("<<NotebookTabChanged>>", self.on_tab_change)
    
    def add_tab(self, _tab_name):
        new_tab = tk.Frame(self.tabControl)
        self.tabs[_tab_name] = new_tab
        self.tabControl.add(new_tab, text=_tab_name, padding=10)
        self.tabControl.pack(expand = 1, fill ="both")
        return new_tab

    def remove_tab(self, _tab_name):
        if _tab_name in self.tabs:
            self.tabControl.forget(self.tabs[_tab_name])
            del self.tabs[_tab_name]
            del self.images[_tab_name]

    def close_current_tab(self):
        if self.tabs:
            tab_name = self.tabControl.tab(self.tabControl.select(), "text")
            self.remove_tab(tab_name)

    def setup_gui_menu(self):
        self.mainMenubar = tk.Menu(self.tkroot)
        self.fileMenu = tk.Menu(self.mainMenubar, tearoff=0)
        self.editMenu = tk.Menu(self.mainMenubar, tearoff=0)
        self.tabMenu = tk.Menu(self.mainMenubar, tearoff=0)
        self.tkroot.config(menu=self.mainMenubar)

        self.mainMenubar.add_cascade(label="File", menu=self.fileMenu)
        self.mainMenubar.add_cascade(label="Edit", menu=self.editMenu)
        self.mainMenubar.add_cascade(label="Tab", menu=self.tabMenu)

        fileOptions = [("Open File", self.open_file), ("Save Image", self.save_image), "separator", ("About", self.show_popup_about), ("Exit", self.tkroot.destroy)]
        editOptions = [("Update Image", self.update_image), ("Deconvolve Image", self.deconvolve_image), ("Close Tab", self.close_current_tab)]
        tabOptions = [("Close Tab", self.close_current_tab)]
        
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
        self.images[img.name] = img
        self.current_tab = img.name

        self.show_image(img.name)

    def update_image(self):
        if self.current_tab not in self.images:
            self.trow_error("No image loaded. Please open an image first.")
            return
        
        kernel = cv2.getGaussianKernel(9, 3) @ cv2.getGaussianKernel(9, 3).T
        kernel = kernel / np.sum(kernel)

        orig_image: CImage = self.images[self.current_tab]
        proc_image = ImageProcessor.apply_kernel_to_image(orig_image, kernel)
        self.images[proc_image.name] = proc_image

        self.show_image(proc_image.name)
    
    def deconvolve_image(self):
        if self.current_tab not in self.images:
            self.trow_error("No image loaded. Please open an image first.")
            return

        orig_image: CImage = self.images[self.current_tab]
        user_input = simpledialog.askstring(title="Kernel Size", prompt="What should be the kernel size?:")
        decon_image, kernel = ImageProcessor.deconvolve_image(orig_image, int(user_input))
        self.images[decon_image.name] = decon_image

        self.show_image(decon_image.name)
    
    def save_image(self):
        assert self.current_tab in self.images, f"save_image: No image of name {self.current_tab} in self.images."

        if self.current_tab not in self.images:
            self.trow_error("No processed image available. Please process an image first.")
            return

        output_path = filedialog.asksaveasfilename(defaultextension=".jpeg")
        if output_path:
            self.images[self.current_tab].save_image(output_path)

    def show_image(self, _image_name):
        assert _image_name in self.images, f"show_image: No image of name {_image_name} in self.images."
        cimage: CImage = self.images[_image_name]
        label_name = cimage.name

        if label_name not in self.tabs:
            self.add_tab(label_name)
        
        if cimage.type == CImage.IMAGE_TYPE.RGB_INT:
            data = cimage.data

        img = Image.fromarray(data)
        photo_img = ImageTk.PhotoImage(image=img)

        img_label = tk.Label(self.tabs[label_name], image=photo_img)
        img_label.image = photo_img
        img_label.pack()
        self.tabControl.select(self.tabs[label_name])

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