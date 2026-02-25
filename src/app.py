import os 
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from src.image_processing import ImageProcessor

class GUI_App:
    def __init__(self, _tkroot):
        self.tabs = {}
        self.images = {}

        self.proc = ImageProcessor()
        self.tkroot = _tkroot

        self.setup_gui_basic()
        self.setup_gui_elements()
        
    def setup_gui_basic(self):
        self.tkroot.title("MUL/ZPO Project")
        self.tkroot.geometry('600x400')

    def setup_gui_elements(self):
        self.setup_gui_menu()

        self.tabControl = ttk.Notebook(self.tkroot)
    
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
        self.tkroot.config(menu=self.mainMenubar)

        self.mainMenubar.add_cascade(label="File", menu=self.fileMenu)
        self.mainMenubar.add_cascade(label="Edit", menu=self.editMenu)

        fileOptions = [("Open File", self.open_file), ("Save Image", self.save_image), "separator", ("About", self.show_popup_about), ("Exit", self.tkroot.destroy)]
        editOptions = [("Update Image", self.update_image), ("Close Tab", self.close_current_tab)]
        
        menuOptions = [(self.fileMenu, fileOptions), (self.editMenu, editOptions)]

        for menu in menuOptions:
            for option in menu[1]:
                if isinstance(option, tuple):
                    menu[0].add_command(label=option[0], command=option[1])
                elif isinstance(option, str):
                    match option:
                        case "separator":
                            menu[0].add_separator()
                        case _:
                            menu[0].add_command(label=option)

    def open_file(self):
        file_path = filedialog.askopenfilename()
        if not file_path or not os.path.isfile(file_path):
            return
        
        self.original_image = self.proc.load_image(file_path)

        self.show_image("Original Image", self.original_image)

    def update_image(self):
        if not hasattr(self, 'original_image'):
            self.trow_error("No image loaded. Please open an image first.")
            return
        
        self.processed_image = self.proc.process_image()

        self.show_image("Processed Image", self.processed_image)
    
    def save_image(self):
        if not hasattr(self, 'processed_image'):
            self.trow_error("No processed image available. Please process an image first.")
            return

        output_path = filedialog.asksaveasfilename(defaultextension=".png")
        if output_path:
            self.proc.save_image(self.processed_image, output_path)

    def show_image(self, _label, _image):
        if _label not in self.tabs:
            self.add_tab(_label)
        im = Image.fromarray(_image)
        self.images[_label] = ImageTk.PhotoImage(image=im)
        label = tk.Label(self.tabs[_label], image=self.images[_label])
        label.pack()
        self.tabControl.select(self.tabs[_label])

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