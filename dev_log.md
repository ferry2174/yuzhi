## 环境变量设置
### 日志
1. TRANSFORMERS_VERBOSITY：设置日志级别，默认是warning，可设置为info、debug、error等。
2. TRANSFORMERS_NO_ADVISORY_WARNINGS: 如果被设置为True，则在使用log.warning_advice()时不会显示警告信息。

### 引擎 - 在 .utils/import_utils.py 文件中设置，目前作用未知
1. USE_TORCH：设置是否使用PyTorch，默认值为 AUTO。
2. USE_ONEFLOW：设置是否使用OneFlow，默认是 AUTO。
3. FORCE_ONEFLOW_AVAILABLE: 设置是否强制使用OneFlow，默认是 AUTO。

## 代码生成
### make
1. make deps_table_update：更新 src/transformers/dependency_versions_table.py 文件。
2. make fix-copies：更新 src/transformers/utils/dummy_{pt|of...}_objects.py 文件。