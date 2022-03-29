import sys
import termcolor
from typing import Optional

try:
    import IPython
except ImportError:
    HAS_IPYTHON = False
else:
    HAS_IPYTHON = IPython.get_ipython() is not None


default_quiet = False


class ConfigurationError(Exception):
    def __init__(
        self, msg: str = "Unknown error", code: int = 1, quiet: Optional[bool] = None
    ):
        if quiet is None:
            quiet = default_quiet
        super().__init__(msg)
        self.code = code
        self.quiet = quiet


if HAS_IPYTHON:
    ipython = IPython.get_ipython()
if HAS_IPYTHON and not hasattr(ipython._showtraceback, "MTJ_ERROR_HOOK_FLAG"):

    def __exception_handler(exception_class, message, traceback):
        if issubclass(exception_class, ConfigurationError) and sys.last_value.quiet:
            print(
                termcolor.colored(f"ERROR {sys.last_value.code}:  {message}", "red"),
                file=sys.stderr,
            )
        else:
            __exception_handler.old_showtraceback(exception_class, message, traceback)

    __exception_handler.old_showtraceback = ipython._showtraceback
    __exception_handler.MTJ_ERROR_HOOK_FLAG = True
    ipython._showtraceback = __exception_handler
