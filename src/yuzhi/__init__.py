__version__ = "0.0.1"

from typing import TYPE_CHECKING

from .utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_oneflow_available,
    is_torch_available,
    logging,
)


logger = logging.get_logger(__name__)

# Base objects, independent of any specific backend
_import_structure = {
    "model.generation": ["activations"],
    "model": [
        "PretrainedConfig",
        "GenerationConfig",
        "PreTrainedModel",
        "PreTrainedTokenizer",
    ],
    "utils" : [
        "get_torch_version",
        "is_bitsandbytes_available",
        "is_torch_available",
        "is_oneflow_available",
        "is_tensor",
        "logging"
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_pt_objects

    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]
else:
    # Importing the actual objects from the backend libraries
    pass

try:
    if not is_oneflow_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_of_objects

    _import_structure["utils.dummy_of_objects"] = [name for name in dir(dummy_of_objects) if not name.startswith("_")]
else:
    # Importing the actual objects from the backend libraries
    pass

# Base objects, independent of any specific backend
if TYPE_CHECKING:
    from .model import (
        BaseModelOutput,
        BaseModelOutputWithPooling,

        GenerationConfig,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
    from .model import activations
    from .utils import (
        OptionalDependencyNotAvailable,
        _LazyModule,
        get_torch_version,
        is_bitsandbytes_available,
        is_oneflow_available,
        is_tensor,
        is_torch_available,
        logging,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_pt_objects import *  # type: ignore
    else:
        pass

    try:
        if not is_oneflow_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_of_objects import *  # type: ignore
    else:
        pass

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
