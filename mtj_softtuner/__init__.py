from ._version import get_versions
from .core import *
from .exceptions import ConfigurationError
from .trainers.basic import BasicTrainer

__version__ = get_versions()["version"]
del get_versions
