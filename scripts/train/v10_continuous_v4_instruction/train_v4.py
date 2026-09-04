"""Register Memory V4 processors, then delegate to the proven V10 trainer."""

from . import dataset_v4 as _dataset_v4  # noqa: F401
from ..v10_continuous.train_v10 import main


if __name__ == "__main__":
    main()

