"""This section has been lifted from the Tkinter Async Execution library and updated with customtkinter and expanded controlls.
The link to the original library is https://github.com/davidhozic/Tkinter-Async-Execute
"""

import asyncio
import sys
from threading import current_thread
from concurrent.futures import Future
from tkinter import messagebox
import customtkinter as ctk
from typing import Coroutine, Optional, Callable, Union, Tuple
from threading import Thread
from concurrent.futures import Future as TFuture


class GLOBAL:
    async_thread: Thread = None
    loop: asyncio.AbstractEventLoop = None


def ctk_par_stop():
    """
    Stops the async queue executor.

    This should be called from tkinter callbacks, not from async functions.
    """
    if GLOBAL.async_thread is None or not GLOBAL.async_thread.is_alive():
        return  # Not running, skip

    loop = GLOBAL.loop
    loop.call_soon_threadsafe(loop.stop)
    GLOBAL.async_thread.join()
    asyncio.set_event_loop(None)
    loop.close()
    ExecutingAsyncWindow.loop = None
    return loop


def ctk_par_start():
    """
    Starts the async queue executor.

    Raises
    ---------
    RuntimeError
        The loop is already running. Stop it with ``tk_async_execute.stop()`` first.
    """
    if GLOBAL.async_thread is not None and GLOBAL.async_thread.is_alive():
        raise RuntimeError(
            "Event loop already started. Stop it with ``tk_async_execute.stop()`` first"
        )

    # Semaphores, etc on version prior to 3.10, call get_event_loop inside, which will cause
    # exceptions if new_event_loop is created and started.
    if sys.version_info.minor < 10:
        loop = asyncio.get_event_loop()
    else:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    GLOBAL.loop = loop
    GLOBAL.async_thread = Thread(target=loop.run_forever)
    ExecutingAsyncWindow.loop = loop
    GLOBAL.async_thread.start()


def tk_execute(method: Callable, *args, **kwargs):
    """
    Allows thread-safe execution of tkinter methods.

    Parameters
    -----------
    method: Callable
        A **tkinter widget** method.
        Methods that are not from tkinter widgets,
        should be called directly without this function.
    args: Any
        Positional arguments to pass to ``method``
    kwargs: Any
        Keyword arguments to pass to ``method``
    """
    widget = method.__self__
    future = TFuture()

    def safe_execute():
        future.set_result(method(*args, **kwargs))

    widget.after_idle(safe_execute)
    return future.result()


def async_execute(
    coro: Coroutine,
    wait: bool = True,
    visible: bool = True,
    pop_up: bool = False,
    callback: Optional[Callable] = None,
    show_exceptions: bool = True,
    message: Optional[Union[str, Callable[[str], str]]] = lambda name: (
        f"Executing {name}"
    ),
    show_stdout: bool = True,
    # Tkinter-specific parameters
    window_title: str = "Async execution window",
    window_resizable: Tuple[bool, bool] = (True, True),
    stdout_label_prefix: str = "> ",
    show_progress_bar: bool = True,
    **kwargs,
):
    window = ExecutingAsyncWindow(
        coro,
        visible,
        pop_up,
        callback,
        show_exceptions,
        message,
        show_stdout,
        window_title,
        window_resizable,
        stdout_label_prefix,
        show_progress_bar,
        **kwargs,
    )
    if wait:
        window.wait_window()

    return window


class ExecutingAsyncWindow(ctk.CTkToplevel):
    loop: asyncio.AbstractEventLoop = None

    def __init__(
        self,
        coro: Coroutine,
        visible: bool = True,
        pop_up: bool = True,
        callback: Optional[Callable] = None,
        show_exceptions: bool = True,
        message: Optional[Union[str, Callable[[str], str]]] = lambda name: (
            f"Executing {name}"
        ),
        show_stdout: bool = True,
        # Tkinter-specific parameters
        window_title: str = "Async execution window",
        window_resizable: Tuple[bool, bool] = (True, True),
        stdout_label_prefix: str = ">",
        show_progress_bar: bool = True,
        **kwargs,
    ):
        loop = self.loop
        if loop is None or not loop.is_running():
            raise RuntimeError("Start the loop first with 'tk_async_execute.start()'")

        super().__init__(**kwargs)
        self.show_exceptions = show_exceptions
        self.title(window_title)
        self.resizable(*window_resizable)
        self.geometry("400x300")
        self.minsize(400, 200)
        frame_main = ctk.CTkFrame(self)
        frame_main.pack(padx=10, pady=10, expand=False, fill="both")

        if callable(message):
            message = message(coro.__name__)
        ctk.CTkLabel(frame_main, text=message).pack(padx=10, pady=10, fill="x")

        if show_progress_bar:
            self.gauge = ctk.CTkProgressBar(frame_main)
            self.gauge.pack(padx=10, pady=10, fill="x", expand=False)
            self.gauge.start()
        else:
            self.gauge = None

        frame_stdout = ctk.CTkFrame(frame_main)
        frame_stdout.pack(padx=10, pady=10, expand=False)

        frame_stdout_acc = ctk.CTkScrollableFrame(self)
        frame_stdout_acc.pack(padx=10, pady=10, expand=True, fill="both", anchor="w")

        self.status_var = ctk.StringVar()
        self.status_var.set("")
        self.status_var_acc = ctk.StringVar()
        self.status_var_acc.set("")
        self.current_thread = current_thread()
        self.old_stdout = sys.stdout

        if show_stdout:
            sys.stdout = self
            ctk.CTkLabel(frame_stdout, text=stdout_label_prefix).grid(row=1, column=0)
            ctk.CTkLabel(frame_stdout, textvariable=self.status_var).grid(
                row=1, column=1
            )
            ctk.CTkLabel(
                frame_stdout_acc,
                textvariable=self.status_var_acc,
                justify="left",
            ).pack(expand=True, fill="x", anchor="w")
        self.protocol("WM_DELETE_WINDOW", lambda: None)

        if not visible:
            self.withdraw()

        if pop_up:
            self.wait_visibility()
            self.grab_set()

        self.awaitable = coro
        self.callback = callback

        self._future = future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        future.add_done_callback(lambda fut: self.after_idle(self.destroy, future))

    @property
    def future(self) -> Future:
        """
        Returns concurrent.futures.Future object.
        This can be used to eg. obtain the coroutine result (``future.result()``)
        or the exception (``future.exception()``).
        """
        return self._future

    def flush(self):
        pass

    def write(self, text: str):
        if current_thread() is not self.current_thread:  # Tkinter thread safety
            self.after_idle(self.write, text)
            return

        if text != "\n":
            self.status_var.set(text)
            self.status_var_acc.set(self.status_var_acc.get() + "\n> " + text)

        # Original sys.stdout can be None when using programs such as pyinstaller (with --noconsole option).
        if self.old_stdout is not None:
            self.old_stdout.write(text)

    def destroy(self, future: asyncio.Future = None) -> None:
        if (
            future is not None
            and (exc := future.exception()) is not None
            and self.show_exceptions
        ):
            # ttkbootstrap compatibility
            title = f"{self.awaitable.__name__} error"
            message = f"{exc}\n\n({type(exc).__name__})"
            if "ttkbootstrap" in sys.modules:
                from ttkbootstrap.dialogs.dialogs import Messagebox

                Messagebox.show_error(message, title, self.master)
            else:
                Messagebox = messagebox.showerror(title, message, master=self.master)

        sys.stdout = self.old_stdout

        if self.callback is not None:
            self.callback()

        return super().destroy()
