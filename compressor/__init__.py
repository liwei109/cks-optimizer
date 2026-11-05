"""cks-compressor: 模型压缩统一框架。

分层结构(自上而下):
- cli: 参数解析与子命令分发(保持薄)
- 领域编排层: quantization / sparsity / prune / search / graph
- 共享能力层: algorithms / observer / datasets / model / eval / kernels
- 适配与导出层: adapter / export
- 支撑: hardware(硬件画像) / utils(通用工具)

依赖铁律:
1. kernels 与 observer 不依赖 adapter —— 由 adapter 单向注册进外部工具
2. model 不依赖 adapter —— 外部工具支持的模型由 adapter 单向注册进来
"""

__version__ = "0.1.0"
