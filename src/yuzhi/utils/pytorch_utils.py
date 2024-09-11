import torch
from packaging import version


parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)

is_torch_greater_or_equal_than_1_13 = parsed_torch_version_base >= version.parse("1.13")
