import tkinter as tk
import customtkinter as ctk
from enum import Enum
from tkinter import messagebox

from src.conv.kernel_generation import KernelGeneration
from src.deconvolution import Deconvolve


class Form:
    class FormType(Enum):
        CONV = 1
        FILTER = 2
        MRK_DECONV = 3
        BRE_DECONV = 4
        TESTING = 5

    def __init__(self, _tkroot, _type: FormType):
        self.tkroot = _tkroot
        self.type = _type

    def popup(self):
        match self.type:
            case Form.FormType.MRK_DECONV:
                decon_form_dialog = MRK_Deconvolution_Form(self.tkroot)
                self.tkroot.wait_window(decon_form_dialog.top)
                if not decon_form_dialog.sent:
                    return (None, None)
                if isinstance(decon_form_dialog.kernel_size, int):
                    return (decon_form_dialog.kernel_size, decon_form_dialog.params)
            case Form.FormType.CONV:
                conv_form = Convolution_Form(self.tkroot)
                self.tkroot.wait_window(conv_form.top)
                if not conv_form.sent:
                    return None
                if (
                    conv_form.kernel_size is not None
                    or conv_form.kernel_sigma is not None
                    or conv_form.kernel_type is not None
                ):
                    return (
                        conv_form.kernel_type,
                        conv_form.kernel_size,
                        conv_form.kernel_sigma,
                    )
                return None
            case Form.FormType.FILTER:
                filter_form = Filter_Form(self.tkroot)
                self.tkroot.wait_window(filter_form.top)
                if not filter_form.sent:
                    return None
                if isinstance(filter_form.offset_size, int):
                    return filter_form.offset_size
                return None
            case Form.FormType.TESTING:
                conv_form = Testing_Form(self.tkroot)
                self.tkroot.wait_window(conv_form.top)
                if not conv_form.sent:
                    return None
                if (
                    conv_form.kernel_size is not None
                    or conv_form.kernel_sigma is not None
                    or conv_form.number_cycles is not None
                ):
                    return (
                        conv_form.number_cycles,
                        conv_form.kernel_size,
                        conv_form.kernel_sigma,
                    )
                return None


class MRK_Deconvolution_Form:
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
        self.top.bind("<Return>", lambda event: self.send())

        self.kernel_size_var = ctk.StringVar()
        self.kernel_size_var.set(str(self.kernel_size))

        self.frame = ctk.CTkFrame(self.top)
        self.frame.pack(padx=10, pady=10, fill="both", expand=True)

        entry_frame = ctk.CTkFrame(self.frame)
        self.verbose_var = tk.BooleanVar()
        self.verbose_checkbox = ctk.CTkCheckBox(
            entry_frame, text="Verbose", variable=self.verbose_var
        )
        self.verbose_checkbox.pack(pady=10, padx=10, fill="x")

        self.rank_var = tk.BooleanVar()
        self.rank_checkbox = ctk.CTkCheckBox(
            entry_frame, text="Rank Regularization", variable=self.rank_var
        )
        self.rank_checkbox.toggle()
        self.rank_checkbox.pack(pady=10, padx=10, fill="x")
        entry_frame.pack(pady=10, padx=10, fill="x", side="top")

        entry_frame = ctk.CTkFrame(self.frame)
        entry_label = ctk.CTkLabel(entry_frame, text="Kernel Size:")
        entry_label.pack(pady=10, padx=10, fill="x", side="left")
        self.kernel_size_entry = ctk.CTkEntry(
            entry_frame, textvariable=self.kernel_size_var
        )
        self.kernel_size_entry.pack(pady=10, padx=10, fill="x", side="right")
        entry_frame.pack(pady=10, padx=10, fill="x", side="top")

        entry_frame = ctk.CTkFrame(self.frame)
        self.mySubmitButton = ctk.CTkButton(
            entry_frame, text="Submit", command=self.send
        )
        self.mySubmitButton.pack(side="right", padx=10, pady=10)
        entry_frame.pack(pady=10, padx=10, fill="x", side="bottom")

        self.top.size = self.frame.size
        self.top.update_idletasks()

    def send(self):
        if self.kernel_size_var.get().isdigit():
            self.kernel_size = int(self.kernel_size_var.get())
        else:
            messagebox.showerror(
                "Invalid offset",
                "Invalid offset value of " + self.kernel_size_var.get(),
            )
            self.kernel_size = None
        self.params.verbose = self.verbose_var.get()
        self.params.low_rank_optimalization = self.rank_var.get()
        self.sent = True
        self.top.destroy()


class Convolution_Form:
    def __init__(self, parent):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.params = Deconvolve.MinRankKernel.MinRankKernelParam()
        self.kernel_size = 25
        self.kernel_sigma = 1
        self.sent = False

        self.kernel_type_var = tk.StringVar()
        self.kernel_size_var = tk.StringVar()
        self.kernel_sigma_var = tk.StringVar()
        self.kernel_type_var.set(KernelGeneration.KernelType.GAUSS_BLUR.value)
        self.kernel_size_var.set(str(self.kernel_size))
        self.kernel_sigma_var.set(str(self.kernel_sigma))

        self.top = ctk.CTkToplevel(parent)
        self.top.title("Convolution Options")
        self.top.geometry("400x300")
        self.top.grab_set()
        self.top.focus_set()
        self.top.minsize(400, 200)
        self.top.bind("<Return>", lambda event: self.send())

        self.frame = ctk.CTkFrame(self.top)
        self.frame.pack(padx=10, pady=10, fill="both", expand=True)

        entry_frame = ctk.CTkFrame(self.frame)
        entry_label = ctk.CTkLabel(entry_frame, text="Kernel Type:")
        entry_label.pack(pady=10, padx=10, fill="x", side="left")
        self.kernel_type_entry = ctk.CTkOptionMenu(
            entry_frame,
            values=list(map(lambda c: c.value, KernelGeneration.KernelType)),
            variable=self.kernel_type_var,
        )
        self.kernel_type_entry.pack(pady=10, padx=10, fill="x", side="right")
        entry_frame.pack(pady=10, padx=10, fill="x", side="top")

        entry_frame = ctk.CTkFrame(self.frame)
        entry_label = ctk.CTkLabel(entry_frame, text="Kernel Size:")
        entry_label.pack(pady=10, padx=10, fill="x", side="left")
        self.kernel_size_entry = ctk.CTkEntry(
            entry_frame, textvariable=self.kernel_size_var
        )
        self.kernel_size_entry.pack(pady=10, padx=10, fill="x", side="right")
        entry_frame.pack(pady=10, padx=10, fill="x", side="top")

        entry_frame = ctk.CTkFrame(self.frame)
        entry_label = ctk.CTkLabel(entry_frame, text="Kernel Sigma:")
        entry_label.pack(pady=10, padx=10, fill="x", side="left")
        self.kernel_sigma_entry = ctk.CTkEntry(
            entry_frame, textvariable=self.kernel_sigma_var
        )
        self.kernel_sigma_entry.pack(pady=10, padx=10, fill="x", side="right")
        entry_frame.pack(pady=10, padx=10, fill="x", side="top")

        entry_frame = ctk.CTkFrame(self.frame)
        self.mySubmitButton = ctk.CTkButton(
            entry_frame, text="Submit", command=self.send
        )
        self.mySubmitButton.pack(side="right", padx=10, pady=10)
        entry_frame.pack(pady=10, padx=10, fill="x", side="bottom")

        self.top.size = self.frame.size
        self.top.update_idletasks()

    def send(self):
        self.kernel_type = self.kernel_type_var.get()
        if self.kernel_size_var.get().isdigit():
            self.kernel_size = int(self.kernel_size_var.get())
        else:
            messagebox.showerror(
                "Invalid offset",
                "Invalid offset value of " + self.kernel_size_var.get(),
            )
            self.kernel_size = None

        if self.kernel_sigma_var.get().isdigit():
            self.kernel_sigma = int(self.kernel_sigma_var.get())
        else:
            messagebox.showerror(
                "Invalid offset",
                "Invalid offset value of " + self.kernel_sigma_var.get(),
            )
            self.kernel_sigma = None
        self.sent = True
        self.top.destroy()


class Filter_Form:
    def __init__(self, parent):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.params = Deconvolve.MinRankKernel.MinRankKernelParam()
        self.offset_size = 25
        self.sent = False

        self.top = ctk.CTkToplevel(parent)
        self.top.title("Filter Options")
        self.top.geometry("400x200")
        self.top.grab_set()
        self.top.focus_set()
        self.top.minsize(400, 200)
        self.top.bind("<Return>", lambda event: self.send())

        self.frame = ctk.CTkFrame(self.top)
        self.frame.pack(padx=10, pady=10, fill="both", expand=True)

        entry_frame = ctk.CTkFrame(self.frame)
        self.offset_size_var = tk.StringVar()
        self.offset_size_var.set(self.offset_size)
        entry_label = ctk.CTkLabel(entry_frame, text="Offset Size:")
        entry_label.pack(pady=10, padx=10, fill="x", side="left")
        self.offset_size_entry = ctk.CTkEntry(
            entry_frame, textvariable=self.offset_size_var
        )
        self.offset_size_entry.pack(pady=10, padx=10, fill="x", side="right")
        entry_frame.pack(pady=10, padx=10, fill="x", side="top")

        entry_frame = ctk.CTkFrame(self.frame)
        self.mySubmitButton = ctk.CTkButton(
            entry_frame, text="Submit", command=self.send
        )
        self.mySubmitButton.pack(side="right", padx=10, pady=10)
        entry_frame.pack(pady=10, padx=10, fill="x", side="bottom")

        self.top.size = self.frame.size
        self.top.update_idletasks()

    def send(self):
        if self.offset_size_var.get().isdigit():
            self.offset_size = int(self.offset_size_var.get())
        else:
            messagebox.showerror(
                "Invalid offset",
                "Invalid offset value of " + self.offset_size_var.get(),
            )
            self.offset_size = None
        self.sent = True
        self.top.destroy()


class Testing_Form:
    def __init__(self, parent):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.params = Deconvolve.MinRankKernel.MinRankKernelParam()
        self.number_cycles = 10
        self.kernel_size = 25
        self.kernel_sigma = 1
        self.sent = False

        self.number_cycles_var = tk.StringVar()
        self.kernel_size_var = tk.StringVar()
        self.kernel_sigma_var = tk.StringVar()
        self.number_cycles_var.set(self.number_cycles)
        self.kernel_size_var.set(str(self.kernel_size))
        self.kernel_sigma_var.set(str(self.kernel_sigma))

        self.top = ctk.CTkToplevel(parent)
        self.top.title("Convolution Options")
        self.top.geometry("400x300")
        self.top.grab_set()
        self.top.focus_set()
        self.top.minsize(400, 200)
        self.top.bind("<Return>", lambda event: self.send())

        self.frame = ctk.CTkFrame(self.top)
        self.frame.pack(padx=10, pady=10, fill="both", expand=True)

        entry_frame = ctk.CTkFrame(self.frame)
        entry_label = ctk.CTkLabel(entry_frame, text="Number of cycles:")
        entry_label.pack(pady=10, padx=10, fill="x", side="left")
        self.kernel_size_entry = ctk.CTkEntry(
            entry_frame, textvariable=self.number_cycles_var
        )
        self.kernel_size_entry.pack(pady=10, padx=10, fill="x", side="right")
        entry_frame.pack(pady=10, padx=10, fill="x", side="top")

        entry_frame = ctk.CTkFrame(self.frame)
        entry_label = ctk.CTkLabel(entry_frame, text="Kernel Size:")
        entry_label.pack(pady=10, padx=10, fill="x", side="left")
        self.kernel_size_entry = ctk.CTkEntry(
            entry_frame, textvariable=self.kernel_size_var
        )
        self.kernel_size_entry.pack(pady=10, padx=10, fill="x", side="right")
        entry_frame.pack(pady=10, padx=10, fill="x", side="top")

        entry_frame = ctk.CTkFrame(self.frame)
        entry_label = ctk.CTkLabel(entry_frame, text="Kernel Sigma:")
        entry_label.pack(pady=10, padx=10, fill="x", side="left")
        self.kernel_sigma_entry = ctk.CTkEntry(
            entry_frame, textvariable=self.kernel_sigma_var
        )
        self.kernel_sigma_entry.pack(pady=10, padx=10, fill="x", side="right")
        entry_frame.pack(pady=10, padx=10, fill="x", side="top")

        entry_frame = ctk.CTkFrame(self.frame)
        self.mySubmitButton = ctk.CTkButton(
            entry_frame, text="Submit", command=self.send
        )
        self.mySubmitButton.pack(side="right", padx=10, pady=10)
        entry_frame.pack(pady=10, padx=10, fill="x", side="bottom")

        self.top.size = self.frame.size
        self.top.update_idletasks()

    def send(self):
        if self.number_cycles_var.get().isdigit():
            self.number_cycles = int(self.number_cycles_var.get())
        else:
            messagebox.showerror(
                "Invalid offset",
                "Invalid offset value of " + self.number_cycles_var.get(),
            )
            self.number_cycles = None

        if self.kernel_size_var.get().isdigit():
            self.kernel_size = int(self.kernel_size_var.get())
        else:
            messagebox.showerror(
                "Invalid offset",
                "Invalid offset value of " + self.kernel_size_var.get(),
            )
            self.offset_size = None

        if self.kernel_sigma_var.get().isdigit():
            self.kernel_sigma = int(self.kernel_sigma_var.get())
        else:
            messagebox.showerror(
                "Invalid offset",
                "Invalid offset value of " + self.kernel_sigma_var.get(),
            )
            self.kernel_sigma = None
        self.sent = True
        self.top.destroy()
