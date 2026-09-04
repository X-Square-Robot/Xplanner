"""Register Memory V3 processors, then delegate to the proven V10 trainer."""

from . import dataset as _dataset  # noqa: F401
from ..pipeline.training import main


if __name__ == "__main__":
    main()
