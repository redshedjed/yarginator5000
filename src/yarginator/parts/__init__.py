"""Part generators. Importing this package registers every built-in generator in ``REGISTRY``."""
from . import bass, drums, instruments, pro_keys, structure, vocals, vocals_scratch  # noqa: F401  (imported for registration)
from .base import REGISTRY, PartCharter, PartContext, PartResult, get_charter, register

__all__ = ["REGISTRY", "PartCharter", "PartContext", "PartResult", "get_charter", "register"]
