from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_oneflow_available,
    is_torch_available,
    logging,
)


logger = logging.get_logger(__name__)

# Base objects, independent of any specific backend
_import_structure = {
    "configuration_utils": [
        "GenerationConfig"
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_pt_objects import *  # type: ignore
else:
    # load torch objects
    pass

try:
    if not is_oneflow_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_of_objects import *  # type: ignore
else:
    # load oneflow objects
    pass

# Base objects, independent of any specific backend
if TYPE_CHECKING:
    from ...utils import (
        OptionalDependencyNotAvailable,
        _LazyModule,
        is_oneflow_available,
        is_torch_available,
        logging,
    )
    from .configuration import (
        GenerationConfig,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_pt_objects import *  # type: ignore
    else:
        # load torch objects
        pass

    try:
        if not is_oneflow_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_of_objects import *  # type: ignore
    else:
        # load oneflow objects
        pass

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
