import numpy as np
from pathlib import Path


def get_n_rows(path: str | Path):
    chunk = 1024*1024   # Process 1 MB at a time.
    f = np.memmap(path)
    num_newlines = sum(np.sum(f[i:i+chunk] == ord('\n'))
                       for i in range(0, len(f), chunk))
    del f
    return num_newlines
