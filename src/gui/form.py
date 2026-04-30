import tkinter as tk
import customtkinter as ctk
from enum import Enum

from src.deconvolution import Deconvolve


class Form:
    class FormType(Enum):
        CONV = 1
        DECONV = 2

    def __init__(self, _tkroot, _type: FormType):
        self.tkroot = _tkroot
        self.type = _type

    def popup(self):
        decon_input_dialog = DeconvolutionForm(self.tkroot)
        self.tkroot.wait_window(decon_input_dialog.top)
        if not decon_input_dialog.sent:
            return (None, None)
        if isinstance(decon_input_dialog.kernel_size, int):
            return (decon_input_dialog.kernel_size, decon_input_dialog.params)


class DeconvolutionForm:
    def __init__(self, parent):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.params = Deconvolve.MinRankKernel.MinRankKernelParam()
        self.kernel_size = 9
        self.sent = False

        self.top = ctk.CTkToplevel(parent)
        self.top.title("Deconvolution Options")
        self.top.geometry("600x400")
        self.top.grab_set()
        self.top.focus_set()
        self.top.minsize(400, 200)

        self.frame = ctk.CTkFrame(self.top)

        entry_frame = ctk.CTkFrame(self.frame)
        self.verbose_var = tk.BooleanVar()
        self.verbose_checkbox = ctk.CTkCheckBox(
            entry_frame, text="Verbose", variable=self.verbose_var
        )
        self.verbose_checkbox.pack(pady=10, padx=10, fill="x", expand=True)

        self.rank_var = tk.BooleanVar()
        self.rank_checkbox = ctk.CTkCheckBox(
            entry_frame, text="Rank Regularization", variable=self.rank_var
        )
        self.rank_checkbox.toggle()
        self.rank_checkbox.pack(pady=10, padx=10, fill="x", expand=True)
        entry_frame.pack(pady=10, padx=10, fill="x", expand=True)

        entry_frame = ctk.CTkFrame(self.frame)
        entry_label = ctk.CTkLabel(entry_frame, text="Kernel Size:")
        entry_label.pack(pady=10, padx=10, fill="x", expand=True, side="left")
        self.kernel_size_entry = ctk.CTkEntry(entry_frame)
        self.kernel_size_entry.insert(0, "9")
        self.kernel_size_entry.pack(
            pady=10, padx=10, fill="x", expand=True, side="right"
        )
        entry_frame.pack(pady=10, padx=10, fill="x", expand=True)

        self.mySubmitButton = ctk.CTkButton(
            self.frame, text="Submit", command=self.send
        )
        self.mySubmitButton.pack()
        self.frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.top.size = self.frame.size
        self.top.update_idletasks()

    def send(self):
        self.kernel_size = int(self.kernel_size_entry.get())
        self.params.verbose = self.verbose_var.get()
        self.params.low_rank_optimalization = self.rank_var.get()
        self.sent = True
        self.top.destroy()
