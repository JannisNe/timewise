from pathlib import Path
import os


def expand(path: Path | str) -> Path:
    """
    Fully expand and resolve the Path with the given environment variables.
    """
    path = Path(path)
    return Path(os.path.expandvars(path)).expanduser().resolve()
