from .dependency_versions_table import deps
from .utils.versions import require_version, require_version_core


# define which module versions we always want to check at run time
# (usually the ones defined in `install_requires` in setup.py)
#
# order specific notes:
# - tqdm must be checked before tokenizers
pkgs_to_check_at_runtime = [
    "python",
    "numpy",
    "pandas",
    "tqdm",
    "torch",
    "safetensors",
    "tokenizers",
]

for pkg in pkgs_to_check_at_runtime:
    if pkg in deps:
        require_version_core(deps[pkg])
    else:
        raise ValueError(f"can't find {pkg} in {deps.keys()}, check dependency_versions_table.py")


def dep_version_check(pkg, hint=None):
    require_version(deps[pkg], hint)
