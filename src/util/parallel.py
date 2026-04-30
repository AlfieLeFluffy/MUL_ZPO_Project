import asyncio
import threading


class Parallel:
    @staticmethod
    def _run_async_task(_loop: asyncio.AbstractEventLoop, _function):
        threading.Thread(target=_loop.run_until_complete, args=(_function,)).start()
