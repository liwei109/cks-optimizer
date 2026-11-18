"""Dataset processing built on pluggable dataset handlers.

Each supported dataset is a ``DatasetHandler`` subclass registered via the
``register`` decorator; see ``handler.py`` for the interface and
``registry.py`` for the built-in ImageNet/COCO/VOC handlers.
"""

from .processing import ProcessedData, data_process

__all__ = ["ProcessedData", "data_process"]
