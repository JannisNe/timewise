import sys
from queue import Queue
from threading import Thread, Event
from typing import Any, Optional, Callable, Mapping


class ErrorQueue(Queue):
    """Queue subclass whose join() re-raises exceptions from worker threads."""

    def __init__(self, stop_event: Event, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_queue = Queue()
        self.stop_event = stop_event

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
                    self.stop_event.set()
                    raise exc_info[1].with_traceback(exc_info[2])
                self.all_tasks_done.wait()

    def raise_errors(self):
        """
        Raise the first worker exception, if any.
        """
        if not self.error_queue.empty():
            exc_info = self.error_queue.get()
            raise exc_info[1].with_traceback(exc_info[2])


class ExceptionSafeThread(Thread):
    """Thread subclass that reports uncaught exceptions to the ErrorQueue."""

    def __init__(
        self,
        error_queue: Any,
        *,
        group: Optional[Any] = None,
        target: Optional[Callable[..., Any]] = None,
        name: Optional[str] = None,
        args: tuple = (),
        kwargs: Optional[Mapping[str, Any]] = None,
        daemon: Optional[bool] = None,
    ):
        super().__init__(
            group=group,
            target=target,
            name=name,
            args=args,
            kwargs=kwargs,
            daemon=daemon,
        )
        self.error_queue = error_queue

    def run(self):
        try:
            super().run()
        except Exception:
            self.error_queue.report_error(sys.exc_info())
