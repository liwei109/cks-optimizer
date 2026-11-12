"""CKS optimizer package."""

import logging

__version__ = "0.1.0"

# Library code must not configure logging; attach a NullHandler so callers that
# never call logging.basicConfig() don't see "No handlers could be found".
logging.getLogger(__name__).addHandler(logging.NullHandler())
