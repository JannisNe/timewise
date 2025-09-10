import sys
from queue import Queue
from threading import Thread


class ErrorQueue(Queue):
    """Queue subclass whose join() re-raises exceptions from worker threads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_queue = Queue()

    def report_error(self, exc_info):
        """Called by workers to push an exception into the error queue."""
        self.error_queue.put(exc_info)
        # Also decrement unfinished_tasks, so join() won't block forever
        with self.all_tasks_done:
            self.unfinished_tasks = max(0, self.unfinished_tasks - 1)
            self.all_tasks_done.notify_all()

    def join(self):
        """Wait until all tasks are done, or raise if a worker failed."""
        with self.all_tasks_done:
            while self.unfinished_tasks:
                if not self.error_queue.empty():
                    exc_info = self.error_queue.get()
                    raise exc_info[1].with_traceback(exc_info[2])
                self.all_tasks_done.wait()


class ExceptionSafeThread(Thread):
    """Thread subclass that reports uncaught exceptions to the ErrorQueue."""

    def __init__(self, error_queue: ErrorQueue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_queue = error_queue

    def run(self):
        try:
            super().run()
        except Exception:
            self.error_queue.report_error(sys.exc_info())
