"""Register Memory V3 processors, then delegate to the proven V10 trainer."""

from . import dataset_v3 as _dataset_v3  # noqa: F401
from ..v10_continuous.train_v10 import main


if __name__ == "__main__":
    main()
