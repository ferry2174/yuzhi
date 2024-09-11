from .. import __version__
from .generic import (
    ContextManagers,
    ExplicitEnum,
    PaddingStrategy,
    TensorType,
    is_numpy_array,
    is_oneflow_tensor,
    is_torch_device,
    is_torch_tensor,
    to_py_obj,
)
from .hub import (
    PushToHubMixin,
)
from .import_utils import (
    ENV_VARS_TRUE_AND_AUTO_VALUES,
    ENV_VARS_TRUE_VALUES,
    USE_ONEFLOW,
    USE_TORCH,
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_bitsandbytes_available,
    is_oneflow_available,
    is_safetensors_available,
    is_tokenizers_available,
    is_torch_available,
    requires_backends,
)
from .pytorch_utils import (
    is_torch_greater_or_equal_than_1_13,
)


WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
CONFIG_FILE_NAME = "config.json"
GENERATION_CONFIG_FILE_NAME = "generation_config.json"
SAFETENSORS_WEIGHTS_NAME = "model.safetensors"
SAFETENSORS_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
