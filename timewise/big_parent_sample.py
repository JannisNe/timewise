import gc
import os
import pickle
import threading
import time
import logging

from timewise.parent_sample_base import ParentSampleBase

logger = logging.getLogger(__name__)


class BigParentSampleBase(ParentSampleBase):
    """
    This should not be used. It was a bad idea. The better way would be to implement a
    ParentSample class that splits the sample file in separate files and then maybe use Dask or similar.
    """

    def __init__(self, base_name, keep_file_in_memory=30*60):
        """
        See doc of ParentSampleBase

        :param base_name: base name for storage directories
        :type base_name: str
        :param keep_file_in_memory: time in seconds to keep the parent sample file in the memory, after that gets written to a cache file on disk
        :type keep_file_in_memory: float
        """
        super().__init__(base_name=base_name)

        self._keep_df_in_memory = keep_file_in_memory
        self._time_when_df_was_used_last = time.time()
        self._df = None
        self._cache_file = os.path.join(self.cache_dir, "cache.pkl")
        self._lock_cache_file = False

        self._clean_thread = threading.Thread(target=self._periodically_drop_df_to_disk, daemon=True, name='ParentSampleCleanThread').start()
        self._stop_thread = False

    def _wait_for_unlock_cache_file(self):
        if self._lock_cache_file:
            logger.debug('cache file locked, waiting')
            while self._lock_cache_file:
                pass

        logger.debug('cache file unlocked')

    @property
    def df(self):
        self._time_when_df_was_used_last = time.time()

        if isinstance(self._df, type(None)):

            if os.path.isfile(self._cache_file):
                logger.debug(f'loading from {self._cache_file}')
                self._wait_for_unlock_cache_file()
                self._lock_cache_file = True

                with open(self._cache_file, "rb") as f:
                    self._df = pickle.load(f)

                self._lock_cache_file = False

            else:
                logger.debug(f'No file {self._cache_file}')

        return self._df

    @df.setter
    def df(self, value):
        self._time_when_df_was_used_last = time.time()
        self._df = value

    def _periodically_drop_df_to_disk(self):
        logger.debug(f'starting cleaning thread for parent sample')

        while True:
            if self._stop_thread:
                break

            if (
                    ((time.time() - self._time_when_df_was_used_last) > self._keep_df_in_memory)
                    and not isinstance(self._df, type(None))
            ):
                logger.debug(f'writing DataFrame to {self._cache_file}')
                self._wait_for_unlock_cache_file()
                self._lock_cache_file = True

                with open(self._cache_file, "wb") as f:
                    pickle.dump(self._df, f)

                self._lock_cache_file = False
                self._df = None
                gc.collect()

            time.sleep(self._keep_df_in_memory / 2)

        logger.debug('stopped clean thread')

    def __del__(self):
        if hasattr(self, "_cache_file") and os.path.isfile(self._cache_file):
            logger.debug(f'removing {self._cache_file}')
            os.remove(self._cache_file)

        if hasattr(self, "clean_thread"):
            logger.debug(f'stopping clean thread')
            self._stop_thread = True
            self._clean_thread.join()