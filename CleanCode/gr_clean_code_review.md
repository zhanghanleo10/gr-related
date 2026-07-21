# vLLM-GR 代码质量与架构问题分析

## 1. 文档目的

本文档记录 `vllm-gr` 当前在代码组织、Clean Code、软件工程原则、上游兼容和工程治理方面存在的主要问题。

分析基于以下版本：

- `vllm-project/vllm`：`releases/v0.22.1`
- `vllm-project/vllm-ascend`：`releases/v0.22.1rc`
- `vllm-gr`：当前仓库代码

本文档当前只负责识别和解释问题，不包含完整重构方案。后续重构任务应以本文中的问题分类和优先级为依据。

---

## 2. 总体判断

`vllm-gr` 当前的问题不只是文件过长、命名不清、重复代码等常规 Clean Code 问题，更根本的问题是扩展方式本身。

当前仓库名义上作为 vLLM 的插件或扩展层存在，但实际上通过大量运行时 monkey patch 深度修改了以下核心路径：

- `SamplingParams`
- `EngineArgs`
- OpenAI Serving
- `EngineCoreClient`
- `EngineCoreProc`
- `EngineCoreRequestType`
- `Scheduler`
- `SchedulerOutput`
- `GPUModelRunner`
- `NPUModelRunner`
- Attention Backend Selector
- vLLM-Ascend Attention Backend
- LMCache Adapter

因此，当前代码更接近：

> 针对 vLLM 0.22.1 和 vLLM-Ascend 0.22.1rc 的运行时私有 fork。

而不是：

> 基于稳定扩展点实现的独立 GR 插件。

这带来了四类相互放大的问题：

1. 运行时正确性和全局状态风险；
2. 对上游私有实现的严重耦合；
3. 目录结构无法表达真实职责；
4. 工程质量门禁与实际覆盖范围不一致。

---

## 3. 问题优先级定义

本文采用以下优先级：

| 优先级 | 含义 |
|---|---|
| P0 | 可能影响运行时正确性、全局行为或进程间一致性，应优先修复 |
| P1 | 严重影响可维护性、升级能力和模块边界，应在架构重构阶段解决 |
| P2 | Clean Code、工程治理和工具配置问题，应持续治理 |
| P3 | 文档、命名和局部一致性问题，可结合日常重构处理 |

---

# 4. P0：运行时正确性和全局污染问题

## 4.1 全局禁用 vLLM greedy sampling 参数校验

位置：

```text
vllm_gr/v1/engine/engine_core_patch.py
```

当前实现中存在类似逻辑：

```python
SamplingParams._verify_greedy_sampling = lambda self: None
```

该修改并没有限定在 GR Beam Search 请求范围内，而是在 EngineCore 进程中永久修改整个 `SamplingParams` 类。

### 影响

- 普通 `generate` 请求也会绕过校验；
- OpenAI Serving 中非 GR 请求也可能受到影响；
- 原本应该在参数解析阶段被拒绝的非法参数组合可能进入 Scheduler；
- 错误位置从参数校验阶段延迟到采样或模型执行阶段；
- 问题可能只在特定参数组合下暴露，排查困难。

### 违反原则

- 最小影响原则；
- 封装原则；
- Open-Closed Principle；
- Principle of Least Surprise；
- Fail Fast。

### 问题本质

GR 为了支持自身请求，修改了整个宿主框架的公共行为。这种做法不应作为长期设计存在。

---

## 4.2 补丁加载不是原子的，可能形成半初始化运行时

位置：

```text
vllm_gr/patch.py
```

当前多个补丁通过独立的 `try/except` 应用：

```python
try:
    ...
except Exception as exc:
    logger.debug("Skipping ...")
```

涉及的补丁包括：

- Logprobs；
- OpenAI Beam Search；
- ChatCompletion 参数转换；
- EngineArgs；
- OpenAIServingModels；
- EngineCore；
- Scheduler；
- Worker；
- LMCache；
- Ascend Attention。

### 影响

可能出现以下不一致状态：

- EngineCore 补丁成功，但 EngineCoreClient 补丁失败；
- Scheduler 能生成 Beam 数据，但 ModelRunner 没有安装对应逻辑；
- OpenAI API 已暴露 Beam 接口，但 EngineCore 不认识对应消息；
- 主进程、EngineCore 子进程和 Worker 进程安装了不同补丁集合；
- 某个补丁失败后系统仍继续启动，直到真实请求触发深层错误。

### 违反原则

- Fail Fast；
- 原子初始化原则；
- 显式契约；
- 可预测性；
- 进程间状态一致性。

### 问题本质

当前插件初始化是“尽量多打一些补丁”，而不是“验证整套能力完整可用后再启动”。

---

## 4.3 通过修改 Enum 内部结构扩展 EngineCore 协议

位置：

```text
vllm_gr/v1/engine/engine_core_patch.py
```

当前代码通过修改 Enum 私有结构增加协议类型，例如：

```text
ADD_BATCH = b"\x06"
BEAM_REQUEST_STEP_UPDATE = b"\x07"
```

同时直接操作：

```python
_value2member_map_
_member_map_
_member_names_
```

vLLM 0.22.1 当前已有的 EngineCore 请求协议值包括：

```text
ADD = 0x00
ABORT = 0x01
START_DP_WAVE = 0x02
UTILITY = 0x03
EXECUTOR_FAILED = 0x04
WAKEUP = 0x05
```

### 影响

- GR 自己维护了一套上游不知道的 wire protocol；
- 主进程和子进程必须使用完全相同的补丁加载顺序；
- 上游新增协议值时可能产生直接冲突；
- Python Enum 内部结构不是稳定扩展 API；
- msgspec 解码、spawn 行为和 import 顺序都会影响结果；
- 协议变更需要同时修改 Client、Socket、EngineCore 和测试。

### 违反原则

- 封装原则；
- Stable Dependencies Principle；
- 显式协议设计；
- Open-Closed Principle。

### 问题本质

当前实现把“内部请求类型扩展”隐藏在运行时 Enum 修改中，没有形成独立、可版本化的协议层。

---

## 4.4 完整替换 `EngineCoreProc.process_input_sockets`

位置：

```text
vllm_gr/v1/engine/core.py
vllm_gr/v1/engine/engine_core_patch.py
```

当前不是在明确扩展点注册新消息，而是复制并替换整个上游 Socket 输入循环。

上游方法负责：

- ZMQ poll；
- READY 消息；
- ADD 专用 decoder；
- ABORT 双队列处理；
- tensor IPC；
- 输入队列顺序；
- DP coordinator 通信。

GR 复制该方法主要为了支持：

```text
ADD_BATCH
BEAM_REQUEST_STEP_UPDATE
```

### 影响

以后上游在该方法中增加任何行为，GR 都必须人工同步，例如：

- 新 RequestType；
- 新 decoder；
- shutdown 行为；
- tracing；
- tensor IPC 变化；
- DP coordinator 行为；
- 异常处理变化。

### 违反原则

- DRY；
- Open-Closed Principle；
- Stable Dependencies Principle；
- 不重复实现宿主内部流程。

### 问题本质

这已经不是普通扩展，而是在运行时替换上游核心循环，等价于维护一个隐藏的局部 fork。

---

## 4.5 父进程、EngineCore 和 Worker 的补丁集合缺少一致性校验

插件会在不同进程中重复加载，当前主要通过以下方式避免重复：

- 模块级布尔变量；
- 在上游类上动态添加 `_patched_for_*` 属性；
- 保存 original function；
- 部分补丁支持 `force=True`；
- 部分补丁没有 marker；
- 某些异常只记录日志后继续。

### 影响

- 不同进程可能处于不同补丁版本；
- 某个补丁成功、另一个失败时没有统一能力检查；
- 无法快速判断某个进程到底安装了哪些补丁；
- 测试环境和真实多进程环境行为可能不同。

### 违反原则

- 明确生命周期；
- 可观测性；
- 原子初始化；
- 一致性。

---

# 5. P1：上游接口耦合问题

## 5.1 大量依赖 vLLM 私有 API

典型依赖包括：

```text
selector._cached_get_attn_backend
GPUModelRunner._prepare_inputs
GPUModelRunner._bookkeeping_sync
NPUModelRunner._prepare_inputs
NPUModelRunner._bookkeeping_sync
Request._block_hasher
SamplingParams._verify_greedy_sampling
```

另外还依赖：

- `runner.input_batch` 的内部布局；
- `sampling_metadata` 的内部 Tensor；
- Scheduler 的内部 `requests` 字典；
- KVCacheManager 的内部方法；
- ModelRunner 返回 tuple 的元素数量和顺序；
- 上游对象动态属性可写。

### 影响

- 方法名称未变化但语义变化时，静态检查无法发现；
- 上游返回值顺序变化时，问题只会在运行时暴露；
- 升级错误通常出现在特定 Beam、特定 batch shape 或特定 NPU 路径；
- 单元测试需要大量模拟上游内部实现；
- 很难支持多个 vLLM 版本。

### 违反原则

- Dependency Inversion Principle；
- Stable Dependencies Principle；
- Open-Closed Principle；
- 面向接口编程。

---

## 5.2 Attention Backend 问题判断合理，但补丁层级错误

位置：

```text
vllm_gr/patch.py
```

当前补丁目标：

```python
vllm.v1.attention.selector._cached_get_attn_backend
```

vLLM 已提供 `AttentionBackendEnum.CUSTOM` 和 `register_backend()` 作为第三方后端注册机制。

问题在于 vLLM-Ascend 的平台后端选择逻辑会根据 MLA、Sparse 和 Compress 状态直接返回 Ascend Backend，没有完整保留 CUSTOM Backend 语义。

因此，GR 识别到的兼容问题是真实存在的，但当前处理方式存在问题：

- 修改 vLLM selector 私有缓存函数；
- 复制上游一部分 KV Cache Layout 设置逻辑；
- 补丁放在顶层 `patch.py`；
- 没有形成版本化 Ascend 兼容边界。

### 建议性质的结论

该问题应被视为：

```text
vLLM-Ascend 0.22.1rc 兼容补丁
```

而不是 GR 核心架构的一部分。

---

## 5.3 没有充分使用上游正式配置入口

vLLM 和 vLLM-Ascend 已提供以下配置入口：

- `scheduler_config.scheduler_cls`
- `parallel_config.worker_cls`
- Platform `check_and_update_config()`
- Attention Backend Registry
- General Plugin Entry Point

当前 GR 更多通过 monkey patch 以下方法完成配置注入：

- `EngineArgs.add_cli_args`
- `EngineArgs.from_cli_args`
- `EngineArgs.create_engine_config`

同时大量配置被写入：

```python
additional_config["vllm_gr_*"] = ...
```

### 影响

`additional_config` 已经变成一个无类型、无 Schema 的隐式控制协议：

- 键名分散；
- 没有统一配置模型；
- 没有配置版本；
- 没有集中验证；
- 拼写错误只能运行时发现；
- 多进程重复解析；
- GPU/NPU 逻辑共享字符串 marker。

### 违反原则

- 显式契约；
- 类型安全；
- 单一配置来源；
- 可验证配置。

---

## 5.4 动态向上游对象增加属性

当前会向 Request 动态增加类似字段：

```text
is_beam
beam_width
is_beam_decode
beam_decode_steps
prefix_len
```

也会向 `SchedulerOutput` 动态增加：

```python
output.beam_data = beam_data
```

### 隐藏数据链

```text
Request
  -> Scheduler.schedule()
  -> SchedulerOutput.beam_data
  -> ModelRunner wrapper
  -> InputBatchProxy
  -> Attention metadata
```

### 影响

- IDE 和 mypy 无法完整追踪；
- 字段漏传不会立即失败；
- SchedulerOutput 被复制、重建或序列化时可能丢失字段；
- 上游 dataclass 或 Struct 改动可能中断动态字段；
- 领域协议没有清晰定义。

### 违反原则

- Encapsulation；
- Explicit Contract；
- Interface Segregation Principle；
- 类型安全。

---

## 5.5 ModelRunner 补丁依赖类名判断

当前存在类似逻辑：

```python
if runner_name == "GPUModelRunner":
    ...
elif runner_name == "NPUModelRunner":
    ...
```

### 影响

- 新 Runner 类型无法自动扩展；
- 子类、重命名或代理类可能无法识别；
- 平台能力通过字符串判断，而不是接口或策略对象；
- GPU/NPU 差异分散在多个文件中。

### 违反原则

- Open-Closed Principle；
- Strategy Pattern 使用缺失；
- 面向能力而非类型名称编程。

---

# 6. P1：目录和功能组织问题

## 6.1 Beam Search 功能横跨几乎所有技术层

当前 Beam Search 相关逻辑分散在：

```text
vllm_gr/
├── sampling_params.py
├── logprobs.py
├── logprobs_patch.py
├── patch.py
├── engine/
│   └── arg_utils_gr.py
├── entrypoints/
│   ├── gr.py
│   └── openai/
│       ├── beam_search_patch.py
│       ├── protocol.py
│       ├── serving_engine.py
│       └── serving_models.py
├── v1/
│   ├── engine/
│   │   ├── core.py
│   │   ├── core_client.py
│   │   ├── core_client_patch.py
│   │   ├── engine_core_patch.py
│   │   └── types.py
│   ├── attention/
│   │   └── backends/
│   │       ├── beam_attn.py
│   │       ├── beam_attn_triton.py
│   │       └── merge_tnd_attention_outputs.py
│   └── metrics/
└── utils/
```

理解一次 Beam Step 需要跨越：

1. API 参数；
2. Beam 算法；
3. Catalog；
4. EngineClient；
5. ZMQ 协议；
6. EngineCore；
7. Scheduler；
8. KV Cache；
9. ModelRunner；
10. Attention；
11. Logprobs；
12. Metrics。

### 问题本质

当前目录主要按照“修改了上游哪些位置”组织，而不是按照“GR 自己有哪些能力”组织。

目录表达的是：

> 我们 patch 了 vLLM 的哪些模块。

而不是：

> vllm-gr 有哪些领域能力，这些能力如何协作。

---

## 6.2 领域逻辑、适配逻辑和运行时补丁混层

典型文件：

```text
vllm_gr/entrypoints/openai/serving_engine.py
```

该文件同时承担：

- OpenAI Serving 适配；
- Beam 状态管理；
- Candidate 排序；
- Catalog 约束；
- Request 提交；
- Engine 输出收集；
- Beam 剪枝；
- 错误处理；
- 指标采集；
- Cache 清理。

该文件中的 `beam_search()` 是一个数百行方法，包含完整领域状态机和基础设施调用。

离线路径：

```text
vllm_gr/entrypoints/gr.py
```

又维护了另一套 Beam Search 状态机。

### 违反原则

- Single Responsibility Principle；
- Separation of Concerns；
- High Cohesion；
- Layered Architecture。

---

## 6.3 在线和离线 Beam Search 重复实现并开始分叉

在线与离线路径分别维护：

- Request ID；
- 父子 Beam 映射；
- EOS；
- Length Penalty；
- Candidate 去重；
- Beam Prune；
- 清理；
- 结果转换。

还存在近似重复方法，例如：

```text
_extract_and_dedup_segment
_beam_request_cleanup
```

### 影响

- 修复离线路径不一定修复在线路径；
- 优化在线路径不一定覆盖离线；
- 两套实现行为会逐渐不一致；
- 测试数量和维护成本被迫翻倍。

### 违反原则

- DRY；
- Common Closure Principle；
- 单一领域模型。

---

## 6.4 `engine_core_patch.py` 是典型 God Module

该文件混合了：

- EngineCore 构造补丁；
- Enum 协议扩展；
- 子进程补丁初始化；
- `run_engine_core` 包装；
- Scheduler 补丁；
- Tensor reshape；
- Logprob 重映射；
- InputBatch Proxy；
- Beam 输入映射；
- Beam 输出折叠；
- GPU ModelRunner 补丁；
- NPU ModelRunner 补丁；
- Worker Patch Registry。

### 问题

这些职责位于完全不同的抽象层级：

```text
协议层
进程生命周期层
调度层
模型执行层
Tensor 变换层
平台适配层
领域逻辑层
```

修改 Scheduler 字段传递时，开发者需要同时理解和规避：

- NPU tuple 处理；
- GPU tensor reshape；
- EngineCore 进程入口；
- Enum 注册；
- InputBatch 代理对象。

### 违反原则

- Single Responsibility Principle；
- Separation of Concerns；
- High Cohesion / Low Coupling；
- Common Closure Principle。

---

## 6.5 `beam_attn.py` 是另一个 God Module

该文件包含：

- Attention Metadata；
- Metadata Builder；
- Request segmentation；
- Standard/Beam row mapping；
- Tensor 构造；
- Block table 更新；
- GPU cascade；
- NPU cascade；
- Forward dispatch；
- Backend registration。

其中 Metadata Builder 同时承担：

- SchedulerOutput 解释；
- Beam 领域语义；
- Tensor layout；
- 平台执行准备。

### 问题本质

Beam 业务概念直接耦合 Attention Tensor 组织，缺少中间协议和转换层。

---

## 6.6 平台代码散落，没有形成平台适配边界

NPU 相关逻辑分布在：

```text
vllm_gr/_env_check.py
vllm_gr/patch.py
vllm_gr/engine/arg_utils_gr.py
vllm_gr/utils/npu_adapter.py
vllm_gr/v1/engine/engine_core_patch.py
vllm_gr/v1/attention/backends/beam_attn.py
requirements/npu.txt
```

但仓库中又存在接近空目录：

```text
vllm_gr/ops/npu/
```

### 问题

- 真正的 NPU 适配不在统一平台边界；
- GPU/NPU 差异通过条件分支散落；
- 空目录与真实职责不一致；
- 平台实现难以独立测试；
- 后续增加其他设备时需要复制更多分支。

---

## 6.7 `entrypoints` 反向承载领域模型

例如 Catalog 领域对象位于：

```text
vllm_gr/entrypoints/openai/serving_models.py
```

离线 GRLLM 又从 OpenAI Serving 模块导入 Catalog。

当前依赖方向类似：

```text
Offline Runtime
    -> OpenAI Serving Adapter
        -> Catalog Domain Object
```

合理依赖方向应是：

```text
Beam/Catalog Domain
    <- Offline Adapter
    <- OpenAI Adapter
```

### 违反原则

- Dependency Inversion Principle；
- Layered Architecture；
- 领域模型独立性。

---

## 6.8 `3rd_party` 目录命名破坏正常 Python 包机制

当前存在：

```text
vllm_gr/3rd_party/lmcache/integration/vllm/vllm_v1_adapter.py
```

由于目录名以数字开头，无法正常通过常规 Python import 使用，只能通过文件路径和 `importlib` 动态加载。

### 影响

- mypy 无法正常追踪；
- IDE 跳转失效；
- package discovery 不直观；
- 动态加载逻辑增加；
- 安装包包含规则更难验证；
- 错误只能在运行时发现。

---

## 6.9 存在空目录和虚假架构边界

当前存在近似空目录：

```text
vllm_gr/config/
vllm_gr/models/
vllm_gr/ops/npu/
```

### 问题

这些目录会误导维护者：

- `config/` 存在，但真实配置散落在 `arg_utils_gr.py` 和 `additional_config`；
- `models/` 存在，但没有形成模型扩展边界；
- `ops/npu/` 存在，但 NPU 逻辑实际分散在多个模块。

目录结构应反映当前真实职责，而不是保留未落地的概念占位。

---

## 6.10 `utils` 开始杂物化

当前 `utils` 中包含：

```text
engine_lifecycle.py
npu_adapter.py
reliable_ops.py
```

它们分别属于：

- Engine 生命周期；
- 平台适配；
- 算子容错。

这些模块没有共同的领域概念，只是暂时被放入 `utils`。

### 风险

如果继续发展，`utils` 会成为：

- 无明确所有权；
- 依赖方向混乱；
- 难以测试；
- 难以重构；

的通用杂物目录。

---

# 7. Clean Code 和软件设计原则问题

| 问题 | 代码表现 | 违反原则 |
|---|---|---|
| 超长方法 | 在线和离线 `beam_search()` 分别数百行 | SRP、可读性 |
| God Module | `engine_core_patch.py`、`beam_attn.py` | SRP、高内聚 |
| 重复算法 | 在线和离线维护两套 Beam 状态机 | DRY |
| 隐式状态 | Request、SchedulerOutput 动态字段 | Encapsulation |
| 全局副作用 | 修改 SamplingParams、EngineArgs、ServingModels | Least Surprise |
| 私有 API 耦合 | `_prepare_inputs`、`_bookkeeping_sync` 等 | DIP、OCP |
| 类型伪装 | `InputBatchProxy`、Mock Array | LSP |
| 平台分支 | 根据类名判断 GPU/NPU | OCP |
| 异常吞噬 | 多处 `except Exception` 后继续 | Fail Fast |
| 时间耦合 | 必须按特定顺序安装父/子进程补丁 | Temporal Coupling |
| 字符串配置协议 | `additional_config["vllm_gr_*"]` | Explicit Contract |
| Patch 命名主导 | `patch.py`、`core.py`、`types.py` | Intention-Revealing Names |
| 动态导入 | 路径加载 `3rd_party` 模块 | Static Discoverability |
| 读取产生副作用 | Proxy 属性读取修改底层 Tensor | Command-Query Separation |

---

## 7.1 `InputBatchProxy` 的状态泄漏风险

位置：

```text
vllm_gr/v1/engine/engine_core_patch.py
```

`InputBatchProxy` 通过 `__getattr__` 模拟上游 InputBatch，但在读取某些属性时会修改原对象中的 Tensor 字段。

### 风险

- 属性读取本身产生副作用；
- 同一 ModelRunner 后续步骤可能看到被修改的数据；
- 异常路径可能未恢复；
- CUDA Graph 或 ACL Graph 捕获状态与真实 Batch 不一致；
- 非 Beam 请求可能被 Beam 代理逻辑影响；
- 很难通过类型系统表达真实语义。

该问题当前应被视为高风险设计点。是否已构成稳定复现的运行时 bug，还需要在真实 GPU/NPU 环境中进行集成验证。

---

# 8. P2：工程配置和质量门禁问题

## 8.1 Ruff 配置没有真正对齐 vLLM 和 vLLM-Ascend

`vllm-gr` 当前主要配置：

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
```

但没有明确启用与上游一致的 lint rule 集。

vLLM 0.22.1 启用了：

```text
E
F
UP
B
ISC
SIM
I
G
```

vLLM-Ascend 也启用了：

```text
E
F
UP
B
SIM
I
G
```

### 问题

当前“使用 Ruff”不等于“执行了与上游相同的代码质量检查”。

因此很多问题可能没有被检查：

- import 顺序；
- bugbear 规则；
- simplify；
- logging format；
- pyupgrade；
- 隐式字符串拼接。

---

## 8.2 mypy strict 配置没有覆盖主体代码

`pyproject.toml` 中存在：

```toml
[tool.mypy]
strict = true
```

但自定义 mypy 脚本主要匹配：

```python
FILES = [
    "vllm_gr/*.py",
]
```

该模式不会递归覆盖主要实现：

```text
vllm_gr/v1/**/*.py
vllm_gr/entrypoints/**/*.py
vllm_gr/engine/**/*.py
vllm_gr/utils/**/*.py
```

此外，配置中还存在多组当前仓库并不存在的目录。

### 结论

仓库表面上启用了 strict mypy，但最复杂、最危险的代码大部分没有进入 mypy 检查。

---

## 8.3 Pre-commit 配置存在重复和副作用

`.pre-commit-config.yaml` 中存在重复 hook，例如：

```text
signoff-commit
```

同时 Ruff 检查使用：

```yaml
--fix
```

### 问题

质量门禁更合理的职责是：

- 检查；
- 报错；
- 阻止不合格提交。

而不是在检查过程中自动修改工作区。

自动修复可以保留为开发命令，但不宜与验证型 hook 混在一起。

---

## 8.4 文档与测试脚本已经漂移

README 中记录的测试命令与实际脚本名称不一致，例如：

```text
README: tools/run_tests.sh
实际: tools/run_test.sh
```

同时还存在：

```text
tools/pre_commit/run_tests.sh
```

### 影响

- 新维护者无法确认哪个是标准测试入口；
- 本地和 CI 可能执行不同测试集合；
- 文档无法作为可信运行手册；
- 重构期间容易漏跑关键测试。

---

## 8.5 核心宿主依赖没有形成机器可执行契约

当前 requirements 中说明：

```text
vllm / torch / vllm-ascend are provided by the base environment
```

但代码实际上精确依赖：

- vLLM 0.22.1；
- vLLM-Ascend 0.22.1rc；
- 特定私有方法；
- 特定 Enum 值；
- 特定 ModelRunner 返回结构；
- 特定 torch/NPU 行为。

### 影响

- 安装可以在错误版本环境中成功；
- 不兼容只能在 import 或请求执行时暴露；
- 依赖解析器无法阻止错误组合；
- 部署系统无法自动检查兼容矩阵。

README 中的人工说明不能替代包元数据和启动时兼容性校验。

---

## 8.6 公共依赖混入 Benchmark 和数据处理依赖

公共 requirements 中包含：

```text
pandas
scikit-learn
pyarrow
opencv-python-headless
```

这些依赖可能用于：

- Benchmark；
- 数据集处理；
- 分析脚本；
- 视频输入。

但不一定属于核心 Beam Runtime 的必需依赖。

### 影响

- 安装体积增加；
- ABI 冲突概率增加；
- 与 torch、vLLM、vLLM-Ascend 的依赖求解更复杂；
- Runtime 与 benchmark 无法独立发布；
- 最小部署镜像难以构建。

---

# 9. P2：测试结构问题

当前部分测试文件本身也出现大文件化，例如：

```text
tests/test_beam_request_unit.py
tests/test_onerec_patch.py
```

大量测试通过 Stub 或 Fake Module 模拟 vLLM 内部对象。

### 这类测试可以验证

- monkey patch 是否赋值；
- wrapper 是否被调用；
- 模拟 tuple 是否可以处理；
- 局部算法是否工作。

### 但无法充分验证

- 真实 SchedulerOutput 是否兼容；
- msgspec 跨进程协议是否一致；
- EngineCore spawn 后补丁是否完整；
- GPU/NPU ModelRunner 的真实返回结构；
- ACL Graph / CUDA Graph 下代理对象是否安全；
- vLLM-Ascend 平台初始化顺序；
- 父进程和 Worker 的补丁一致性；
- 与上游小版本升级后的兼容性。

### 问题本质

当前测试更多是在证明：

> GR 的补丁代码在模拟环境中可以运行。

而不是证明：

> GR 与真实 vLLM / vLLM-Ascend 的契约仍然兼容。

---

# 10. 问题严重度汇总

## 10.1 P0：必须优先处理

1. 全局禁用 `SamplingParams._verify_greedy_sampling`；
2. 补丁失败后继续启动，可能形成半初始化系统；
3. 运行时修改 `EngineCoreRequestType` Enum 和 wire protocol；
4. 完整替换 `process_input_sockets`；
5. 主进程、EngineCore 和 Worker 缺少补丁一致性校验；
6. 全局 monkey patch 缺少明确作用域和恢复机制。

## 10.2 P1：决定长期可维护性

1. `engine_core_patch.py` God Module；
2. `beam_attn.py` God Module；
3. 在线和离线 Beam Search 双实现；
4. 动态字段形成隐藏协议；
5. 私有 ModelRunner 和 Scheduler API 耦合；
6. Attention Selector 私有函数补丁；
7. GPU/NPU 分支分散且依赖类名；
8. 目录按上游 patch 点组织，而非按 GR 能力组织；
9. Catalog 领域对象位于 OpenAI Serving 适配层；
10. 平台适配没有形成独立边界；
11. 配置通过 `additional_config` 字符串协议扩散；
12. V1 ModelRunner 强绑定。

## 10.3 P2：工程治理问题

1. 超长方法和高圈复杂度；
2. 空目录与虚假模块边界；
3. `utils` 杂物化；
4. `3rd_party` 非标准 Python 包；
5. mypy 主体代码漏检；
6. Ruff 规则未与上游对齐；
7. 重复 pre-commit hook；
8. README 与脚本名称不一致；
9. 核心、benchmark 和数据依赖没有拆分；
10. 测试大量依赖 Stub，真实集成契约不足。

---

# 11. 根因分析

当前代码问题形成了以下循环：

```text
上游缺少一个扩展点
        ↓
复制或替换整个上游方法
        ↓
通过动态属性传递额外状态
        ↓
父进程、EngineCore、Worker 重复打补丁
        ↓
测试需要大量模拟上游私有对象
        ↓
只能精确锁定上游版本
        ↓
升级时继续增加兼容补丁
```

如果只做以下工作：

- 拆文件；
- 改名称；
- 提取函数；
- 增加 Ruff；
- 减少单文件行数；

并不能从根本上解决问题，只会把同一套高耦合补丁分散到更多文件。

---

# 12. 后续重构必须建立的边界

后续重构至少需要形成以下四个明确边界：

```text
1. Beam / Catalog 领域能力
2. vLLM 公共适配层
3. 版本化兼容与补丁层
4. GPU / NPU 平台实现层
```

其中：

- Beam 算法不应依赖 OpenAI Serving；
- Offline 和 Online 应复用同一领域状态机；
- vLLM 私有兼容逻辑应集中管理并绑定版本；
- GPU/NPU 差异应通过平台策略或接口隔离；
- EngineCore 协议应显式定义，而不是运行时修改 Enum；
- 所有补丁必须具备能力校验、版本校验和失败策略；
- 配置应有明确类型和统一解析入口；
- 测试应区分领域单测、兼容契约测试和真实集成测试。

---

# 13. 当前结论

当前仓库已经具备较强的功能实现能力，但代码结构和扩展方式使其承担了过高的升级与维护成本。

最严重的问题不是代码风格，而是：

- 全局副作用；
- 隐式协议；
- 私有 API 耦合；
- 运行时补丁不一致；
- 目录无法表达真实架构；
- 领域逻辑与基础设施混合。

因此，本次重构应优先解决架构边界和运行时正确性问题，再进行常规 Clean Code 优化。

---

# 14. 上游参考位置

分析时重点对照了以下上游模块：

## vLLM 0.22.1

```text
vllm/v1/engine/core.py
vllm/v1/engine/__init__.py
vllm/v1/core/sched/output.py
vllm/v1/attention/selector.py
vllm/v1/attention/backends/registry.py
vllm/plugins/__init__.py
pyproject.toml
requirements/common.txt
```

仓库：

```text
https://github.com/vllm-project/vllm/tree/releases/v0.22.1
```

## vLLM-Ascend 0.22.1rc

```text
vllm_ascend/platform.py
pyproject.toml
```

仓库：

```text
https://github.com/vllm-project/vllm-ascend/tree/releases/v0.22.1rc
```

---

## 15. 验证限制

本文主要基于静态代码分析和上游源码对照。

以下问题仍建议在真实环境中进一步验证：

- `InputBatchProxy` 是否造成稳定可复现的状态泄漏；
- GPU ModelRunner 返回结构的真实兼容性；
- NPU ModelRunner 和 ACL Graph 路径的真实兼容性；
- 多进程 spawn 后补丁集合的一致性；
- msgspec 和 ZMQ 协议在不同加载顺序下的行为；
- LMCache 补丁与当前 vLLM-Ascend 组合的运行时稳定性；
- 非 Beam 请求是否会受到全局 SamplingParams 修改影响。

