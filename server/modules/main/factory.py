import importlib
import pkgutil

import server.modules.main.processors

from .interfaces import LayoutProcessor
from .models import ModelChoice

PROCESSOR_REGISTORY: dict[ModelChoice, LayoutProcessor] = {}

def register(choice: ModelChoice):
    def decorator(proc: LayoutProcessor):
        PROCESSOR_REGISTORY[choice] = proc
        return proc
    return decorator

for _, modname, _ in pkgutil.iter_modules(server.modules.main.processors.__path__):
    importlib.import_module(f'{server.modules.main.processors.__name__}.{modname}')