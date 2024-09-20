"""
Import utilities: Utilities related to imports and our lazy inits.
"""
import importlib.metadata
import importlib.util
import os
from collections import OrderedDict
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any, Tuple, Union

from packaging import version

from . import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check if the package spec exists and grab its version to avoid importing a local directory
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            # Fallback method: Only for "torch" and versions containing "dev"
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            else:
                # For packages other than "torch", don't attempt the fallback and set as not available
                package_exists = False
        logger.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_ONEFLOW = os.environ.get("USE_ONEFLOW", "AUTO").upper()

# Try to run a native pytorch job in an environment with TorchXLA installed by setting this value to 0.
USE_TORCH_XLA = os.environ.get("USE_TORCH_XLA", "1").upper()

FORCE_ONEFLOW_AVAILABLE = os.environ.get("FORCE_ONEFLOW_AVAILABLE", "AUTO").upper()

_bitsandbytes_available = _is_package_available("bitsandbytes")
_safetensors_available = _is_package_available("safetensors")
_tokenizers_available = _is_package_available("tokenizers")

_torch_version = "N/A"
_torch_available = False
if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_ONEFLOW not in ENV_VARS_TRUE_VALUES:
    _torch_available, _torch_version = _is_package_available("torch", return_version=True)
else:
    logger.info("Disabling PyTorch because USE_ONEFLOW is set")
    _torch_available = False

_torch_xla_available = False
if USE_TORCH_XLA in ENV_VARS_TRUE_VALUES:
    _torch_xla_available, _torch_xla_version = _is_package_available("torch_xla", return_version=True)
    if _torch_xla_available:
        logger.info(f"Torch XLA version {_torch_xla_version} available.")

_of_version = "N/A"
_of_available = False
if FORCE_ONEFLOW_AVAILABLE in ENV_VARS_TRUE_VALUES:
    _of_available = True
else:
    if USE_ONEFLOW in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
        _of_version = importlib.metadata.version("oneflow")
        _of_available = importlib.util.find_spec("oneflow") is not None and _of_version is not None
    else:
        logger.info("Disabling OneFlow because USE_TORCH is set")

def is_torch_available() -> bool:
    return _torch_available

def is_oneflow_available() -> bool:
    return _of_available

def is_bitsandbytes_available():
    if not is_oneflow_available():
        return False
    # bitsandbytes throws an error if cuda is not available
    # let's avoid that by adding a simple check
    import oneflow
    return _bitsandbytes_available and oneflow.cuda.is_available()

def is_safetensors_available():
    return _safetensors_available

def is_tokenizers_available():
    return _tokenizers_available

def get_torch_version():
    return _torch_version

@lru_cache
def is_torch_xla_available(check_is_tpu=False, check_is_gpu=False):
    """
    Check if `torch_xla` is available. To train a native pytorch job in an environment with torch xla installed, set
    the USE_TORCH_XLA to false.
    """
    assert not (check_is_tpu and check_is_gpu), "The check_is_tpu and check_is_gpu cannot both be true."

    if not _torch_xla_available:
        return False

    import torch_xla

    if check_is_gpu:
        return torch_xla.runtime.device_type() in ["GPU", "CUDA"]

    return True

# docstyle-ignore
PYTORCH_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
ONEFLOW_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""

BACKENDS_MAPPING = OrderedDict(
    [
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("oneflow", (is_oneflow_available, ONEFLOW_IMPORT_ERROR))
    ]
)

def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__

    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError("".join(failed))


class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    def __getattribute__(cls, key):
        if key.startswith("_") and key != "_from_config":
            return super().__getattribute__(key)
        requires_backends(cls, cls._backends)

class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))

class OptionalDependencyNotAvailable(BaseException):
    """Internally used error class for signalling an optional dependency was not found."""

def is_torchdynamo_compiling():
    if not is_torch_available():
        return False
    try:
        # Importing torch._dynamo causes issues with PyTorch profiler (https://github.com/pytorch/pytorch/issues/130622) hence rather relying on `torch.compiler.is_compiling()` when possible.
        if version.parse(_torch_version) >= version.parse("2.3.0"):
            import torch

            return torch.compiler.is_compiling()
        else:
            import torch._dynamo as dynamo  # noqa: F401

            return dynamo.is_compiling()
    except Exception:
        return False
