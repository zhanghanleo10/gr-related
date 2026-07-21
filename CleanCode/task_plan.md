# vLLM-GR 重构任务拆分与依赖规划

## 1. 规划目标

本次重构不应只进行文件拆分和命名调整，而应逐步解决以下核心问题：

1. 消除影响普通 vLLM 请求的全局副作用；
2. 建立可靠、可验证的补丁初始化机制；
3. 抽取独立的 Beam Search 领域模型；
4. 统一在线和离线 Beam Search 实现；
5. 显式定义 EngineCore、Scheduler 和 ModelRunner 之间的数据契约；
6. 隔离 vLLM 和 vLLM-Ascend 私有 API；
7. 建立 GPU/NPU 平台适配边界；
8. 重新组织目录，使目录能够表达真实职责；
9. 建立可执行的静态检查、契约测试和集成测试体系。

建议按照以下顺序推进：

```text
稳定当前行为
    ↓
消除 P0 全局风险
    ↓
定义领域和数据契约
    ↓
重构 Engine / Scheduler / Runner 集成
    ↓
隔离 GPU / NPU 平台差异
    ↓
调整目录结构
    ↓
收紧类型检查和质量门禁
    ↓
删除旧补丁和兼容代码
```

不建议一开始就大规模移动目录。目录调整应当发生在职责边界明确之后，否则只是把当前耦合关系搬到新的目录中。

---

## 2. 总体任务依赖图

```text
T0 现状基线与兼容矩阵
│
├── T1 修复全局 SamplingParams 副作用
│
├── T2 建立补丁注册与原子初始化机制
│
├── T3 建立类型化 GR 配置
│
├── T4 抽取 Beam Search 领域模型
│   └── T5 统一在线与离线 Beam 状态机
│
├── T6 定义 EngineCore 扩展协议
│   └── T7 建立 Scheduler Beam 上下文
│       └── T8 建立 ModelRunner Adapter
│
├── T9 隔离 Attention Backend 兼容逻辑
│   └── T10 建立 GPU/NPU 平台策略
│
├── T11 建立测试分层和兼容契约测试
│
└── T12 建立 Ruff、mypy 和 CI 基线

T5 + T8 + T10
        ↓
T13 目录结构重组
        ↓
T14 删除旧补丁、动态字段和重复实现
        ↓
T15 文档、迁移说明和最终质量收口
```

---

## 3. 第一阶段：建立安全基线

### T0：现状基线与兼容矩阵

#### 目标

在开始重构前固定当前行为，明确哪些行为必须保持，哪些行为属于当前缺陷。

#### 工作内容

- 记录支持的运行模式：
  - Offline；
  - OpenAI Serving；
  - GPU；
  - NPU；
  - 单 Engine；
  - 多 Engine；
  - LMCache；
  - Catalog；
  - Beam Attention。
- 建立 vLLM 和 vLLM-Ascend 兼容矩阵；
- 固定当前关键请求的输入输出；
- 增加基础 smoke test；
- 记录当前 monkey patch 清单；
- 记录每个补丁影响的进程：
  - Frontend；
  - EngineCore；
  - Worker；
  - 所有进程。

#### 交付物

```text
docs/refactoring/current_behavior.md
docs/refactoring/compatibility_matrix.md
tests/compatibility/test_current_behavior.py
```

#### 依赖

无。

#### 是否可并行

这是其他任务的共同前置任务，应最先完成。

#### 建议负责人

由熟悉整体架构和主要运行路径的同事负责。

---

### T1：修复全局 SamplingParams 副作用

#### 目标

删除或限制以下全局修改：

```python
SamplingParams._verify_greedy_sampling = lambda self: None
```

#### 工作内容

- 找出 GR Beam Search 需要绕过校验的真实参数组合；
- 将特殊参数转换限制在 GR 请求入口；
- 普通 vLLM 请求继续使用原始校验逻辑；
- 增加普通请求不受 GR 插件影响的回归测试；
- 增加非法 Beam 参数测试。

#### 交付物

- 不再全局替换 `_verify_greedy_sampling`；
- GR 请求拥有独立参数转换或验证逻辑；
- 普通请求回归测试。

#### 依赖

依赖 T0。

#### 是否可并行

可与 T2、T3、T4、T6、T9 并行。

#### 修改冲突风险

主要涉及：

```text
vllm_gr/v1/engine/engine_core_patch.py
vllm_gr/sampling_params.py
OpenAI 请求参数转换模块
```

应避免与 T2 同时修改同一段补丁初始化代码。

---

### T2：建立补丁注册与原子初始化机制

#### 目标

把零散的 monkey patch 调用统一成可验证、可观测的补丁系统。

#### 工作内容

定义统一补丁描述：

```python
class PatchSpec:
    name: str
    target: str
    supported_versions: VersionRange
    required_processes: set[ProcessRole]
    required: bool
    apply: Callable
    validate: Callable
```

统一处理：

- 补丁名称；
- 支持版本；
- 前置条件；
- 幂等性；
- 是否为必需补丁；
- 补丁应用结果；
- 补丁验证；
- 失败策略；
- 进程角色；
- 启动日志。

建议补丁状态至少包括：

```text
NOT_APPLIED
APPLIED
SKIPPED
FAILED
INCOMPATIBLE
```

必需补丁失败时，应阻止服务继续启动，不应只记录 debug 日志。

#### 交付物

```text
vllm_gr/compat/patch_registry.py
vllm_gr/compat/patch_result.py
vllm_gr/compat/version_check.py
```

#### 依赖

依赖 T0。

#### 是否可并行

可与领域和平台任务并行，但后续所有兼容补丁都应迁移到该机制。

#### 修改冲突风险

高风险文件：

```text
vllm_gr/patch.py
vllm_gr/v1/engine/engine_core_patch.py
```

建议该任务与 T6 的负责人明确分工，或者由同一人顺序完成。

---

### T3：建立类型化 GR 配置

#### 目标

停止把 `additional_config` 当作无类型字符串字典使用。

#### 工作内容

建立统一配置模型，例如：

```python
@dataclass
class GRConfig:
    enabled: bool
    catalog_path: str | None
    attention_backend: str | None
    beam: BeamConfig
    compatibility: CompatibilityConfig
```

提供统一入口：

```python
GRConfig.from_vllm_config(vllm_config)
```

要求：

- 所有键名集中定义；
- 默认值集中定义；
- 启动时统一验证；
- 配置错误尽早失败；
- 其他模块不直接读取 `additional_config["vllm_gr_*"]`；
- 配置模型可在多进程中稳定序列化。

#### 交付物

```text
vllm_gr/config/models.py
vllm_gr/config/parser.py
vllm_gr/config/validation.py
```

#### 依赖

依赖 T0。

#### 是否可并行

可独立推进。

#### 修改冲突风险

涉及：

```text
vllm_gr/engine/arg_utils_gr.py
vllm_gr/patch.py
vllm_gr/_env_check.py
```

---

## 4. 第二阶段：抽取 Beam Search 领域能力

### T4：抽取 Beam Search 领域模型

#### 目标

将 Beam Search 核心算法从 OpenAI Serving 和 Offline 入口中抽离。

#### 工作内容

抽取以下领域对象：

```text
BeamState
BeamCandidate
BeamGroup
BeamStepInput
BeamStepResult
BeamFinishedReason
BeamRequestContext
```

抽取以下纯领域操作：

- Candidate 扩展；
- Candidate 排序；
- Beam prune；
- EOS 判断；
- Length penalty；
- 父子 Beam 映射；
- Segment 去重；
- Finished Beam 管理；
- 最终结果排序。

要求领域层：

- 不依赖 AsyncLLM；
- 不依赖 OpenAI 协议；
- 不依赖 Scheduler；
- 不依赖 GPU/NPU；
- 不依赖 ZMQ；
- 尽量使用纯 Python 数据结构；
- 可以通过纯单元测试验证。

#### 交付物

```text
vllm_gr/beam/domain.py
vllm_gr/beam/ranking.py
vllm_gr/beam/pruning.py
vllm_gr/beam/state_machine.py
tests/beam/domain/
```

#### 依赖

依赖 T0。

#### 是否可并行

可与 T6、T9、T11 并行。

#### 修改冲突风险

主要涉及：

```text
vllm_gr/entrypoints/gr.py
vllm_gr/entrypoints/openai/serving_engine.py
```

在 T4 完成前，不建议其他任务大规模修改这两个文件。

---

### T5：统一在线和离线 Beam Search 状态机

#### 目标

删除在线和离线两套 Beam Search 算法实现。

#### 工作内容

在线和离线入口只保留适配职责：

```text
输入协议转换
    ↓
调用共享 Beam 状态机
    ↓
提交 Engine 请求
    ↓
转换输出协议
```

建议定义执行端口：

```python
class BeamEnginePort(Protocol):
    def add_requests(...): ...
    def update_beam_step(...): ...
    def collect_outputs(...): ...
    def abort(...): ...
```

实现：

```text
OfflineBeamEngineAdapter
AsyncBeamEngineAdapter
```

两者复用同一个：

```text
BeamSearchCoordinator
```

#### 交付物

```text
vllm_gr/beam/coordinator.py
vllm_gr/adapters/offline/beam_engine.py
vllm_gr/adapters/openai/beam_engine.py
```

#### 依赖

依赖 T4。

建议等待 T6 的协议对象基本稳定后再完成最终集成。

#### 是否可并行

领域状态机完成后，Offline Adapter 和 OpenAI Adapter 可由不同同事并行实现。

#### 修改冲突风险

高风险文件：

```text
vllm_gr/entrypoints/gr.py
vllm_gr/entrypoints/openai/serving_engine.py
```

建议将文件所有权分开：

- 一名同事负责共享 Coordinator；
- 一名同事负责 Offline Adapter；
- 一名同事负责 OpenAI Adapter。

---

## 5. 第三阶段：显式化 Engine、Scheduler 和 Runner 契约

### T6：定义 EngineCore 扩展协议

#### 目标

将当前运行时修改 Enum 的隐式协议改为显式、可版本化的 GR 协议。

#### 工作内容

定义独立消息对象：

```text
GRAddBatchRequest
GRBeamStepUpdateRequest
GRRequestEnvelope
GRResponseEnvelope
GRProtocolVersion
```

评估两种实现方向：

#### 方向 A：复用上游 Utility Request

如果性能和调用模型允许，优先使用已有 Utility 通道传递 GR 控制消息。

#### 方向 B：建立 GR 独立 Envelope

如果必须走高性能请求通道，则使用 GR 自己的消息 Envelope，不再修改上游 Enum 内部状态。

要求：

- 明确协议版本；
- 明确消息序列化方式；
- 明确未知消息处理；
- 明确客户端与 EngineCore 的能力握手；
- 启动时校验协议版本；
- 不依赖修改 `_member_map_` 等 Enum 内部属性。

#### 交付物

```text
vllm_gr/protocol/messages.py
vllm_gr/protocol/version.py
vllm_gr/protocol/codec.py
tests/protocol/
```

#### 依赖

依赖 T0。

#### 是否可并行

可与 T4、T9 并行。

#### 修改冲突风险

主要涉及：

```text
vllm_gr/v1/engine/core.py
vllm_gr/v1/engine/core_client.py
vllm_gr/v1/engine/core_client_patch.py
vllm_gr/v1/engine/engine_core_patch.py
```

该任务应由熟悉 EngineCore、ZMQ 和 msgspec 的同事负责。

---

### T7：建立 Scheduler Beam 上下文

#### 目标

删除向 Request 和 SchedulerOutput 动态添加 Beam 属性的做法。

#### 工作内容

定义显式对象：

```python
@dataclass
class BeamRequestMetadata:
    beam_width: int
    beam_decode_steps: int
    prefix_len: int
    parent_request_id: str | None
```

```python
@dataclass
class BeamSchedulerContext:
    new_requests: ...
    cached_requests: ...
    beam_mappings: ...
```

需要明确：

- 哪些字段属于请求生命周期；
- 哪些字段属于单次调度；
- 哪些字段需要跨进程传输；
- 哪些字段只供 Attention 使用；
- 哪些字段应进入 ModelRunner；
- 哪些字段不应该污染上游 Request。

可通过 Adapter 将 GR 上下文与上游 SchedulerOutput 组合：

```python
@dataclass
class GRSchedulerStep:
    upstream: SchedulerOutput
    beam: BeamSchedulerContext | None
```

#### 交付物

```text
vllm_gr/scheduler/context.py
vllm_gr/scheduler/adapter.py
vllm_gr/scheduler/request_metadata.py
```

#### 依赖

依赖 T6。

最好同时参考 T4 中的领域对象，避免重复定义 Beam 概念。

#### 是否可并行

Request 生命周期元数据和单次调度上下文可以拆给不同同事，但最终接口必须统一评审。

#### 修改冲突风险

主要涉及：

```text
vllm_gr/v1/engine/engine_core_patch.py
Scheduler patch
Request 动态字段
SchedulerOutput.beam_data
```

---

### T8：建立 ModelRunner Adapter

#### 目标

停止直接在大文件中替换 GPU/NPU ModelRunner 私有方法。

#### 工作内容

定义统一 Runner Adapter：

```python
class BeamModelRunnerAdapter(Protocol):
    def prepare_beam_inputs(...): ...
    def update_after_execution(...): ...
    def collapse_beam_outputs(...): ...
```

分别实现：

```text
GPUBeamModelRunnerAdapter
NPUBeamModelRunnerAdapter
```

Adapter 负责：

- Runner 输入映射；
- Beam batch reshape；
- Sampling metadata 调整；
- 输出折叠；
- Logprob 重映射；
- 上游版本差异处理。

要求：

- 不根据类名字符串判断平台；
- 不在属性读取过程中修改原对象；
- 明确记录被修改的 Tensor；
- 必要时使用上下文管理器保证恢复；
- 将上游私有 API 访问限制在 Adapter 内部。

#### 交付物

```text
vllm_gr/runtime/model_runner/base.py
vllm_gr/runtime/model_runner/gpu.py
vllm_gr/runtime/model_runner/npu.py
vllm_gr/runtime/model_runner/compat_v0221.py
```

#### 依赖

依赖 T7。

#### 是否可并行

GPU Adapter 和 NPU Adapter 可以由不同同事并行实现。

#### 修改冲突风险

旧实现集中在：

```text
vllm_gr/v1/engine/engine_core_patch.py
```

建议先定义 Base Protocol，再分别迁移 GPU/NPU。

---

## 6. 第四阶段：平台和 Attention 隔离

### T9：隔离 Attention Backend 兼容逻辑

#### 目标

将 vLLM-Ascend CUSTOM Backend 兼容问题从顶层 `patch.py` 中移出。

#### 工作内容

建立明确的版本兼容模块：

```text
vllm_gr/compat/vllm/v0_22_1/
vllm_gr/compat/vllm_ascend/v0_22_1rc/
```

该模块负责：

- 检查上游版本；
- 检查 Ascend 是否已原生支持 CUSTOM Backend；
- 只在确实需要时安装兼容补丁；
- 兼容补丁失效时提供明确错误；
- 保留上游 KV Cache Layout 行为；
- 增加针对 Backend 选择的契约测试。

同时评估是否可以：

- 直接使用 `register_backend()`；
- 通过 NPUPlatform 配置完成；
- 向 vLLM-Ascend 上游提交修复。

#### 交付物

```text
vllm_gr/compat/vllm_ascend/v0_22_1rc/attention_backend.py
tests/compatibility/test_ascend_attention_backend.py
```

#### 依赖

依赖 T0。

后续应接入 T2 的补丁注册系统。

#### 是否可并行

可独立推进。

#### 修改冲突风险

涉及：

```text
vllm_gr/patch.py
vllm_gr/v1/attention/backends/beam_attn.py
```

---

### T10：建立 GPU/NPU 平台策略

#### 目标

将分散的 GPU/NPU 条件分支集中到平台层。

#### 工作内容

定义平台接口：

```python
class GRPlatform(Protocol):
    def create_model_runner_adapter(...): ...
    def create_attention_backend(...): ...
    def validate_config(...): ...
    def prepare_runtime(...): ...
```

实现：

```text
GPUPlatform
NPUPlatform
```

平台层负责：

- Runner Adapter 选择；
- Attention Backend 选择；
- 平台配置验证；
- 环境检查；
- 平台专用算子；
- 平台专用依赖；
- 平台能力声明。

清理以下模式：

```python
if runner_name == "GPUModelRunner":
    ...
elif runner_name == "NPUModelRunner":
    ...
```

#### 交付物

```text
vllm_gr/platform/base.py
vllm_gr/platform/gpu.py
vllm_gr/platform/npu.py
```

#### 依赖

依赖 T8 和 T9。

#### 是否可并行

平台接口确定后，GPU 和 NPU 实现可以并行。

#### 修改冲突风险

涉及：

```text
vllm_gr/_env_check.py
vllm_gr/utils/npu_adapter.py
vllm_gr/v1/attention/backends/beam_attn.py
vllm_gr/v1/engine/engine_core_patch.py
```

---

## 7. 第五阶段：工程质量和测试体系

### T11：建立测试分层和兼容契约测试

#### 目标

将当前以 Stub 为主的测试拆成不同层级。

#### 测试分层

##### 第一层：领域单元测试

覆盖：

- Beam 排序；
- Beam prune；
- EOS；
- Length penalty；
- Candidate 去重；
- 状态迁移。

不依赖 vLLM、GPU 或 NPU。

##### 第二层：Adapter 契约测试

覆盖：

- Offline Adapter；
- OpenAI Adapter；
- Scheduler Adapter；
- ModelRunner Adapter；
- Attention Backend Adapter。

可以使用少量 Fake，但必须围绕明确接口。

##### 第三层：上游兼容测试

针对固定版本运行：

- vLLM 0.22.1；
- vLLM-Ascend 0.22.1rc。

检查：

- 方法签名；
- 返回结构；
- Enum 值；
- SchedulerOutput 字段；
- Runner 私有方法是否仍存在；
- CUSTOM Backend 行为。

##### 第四层：真实集成测试

覆盖：

- Offline GPU；
- Online GPU；
- Offline NPU；
- Online NPU；
- 多 Engine；
- EngineCore spawn；
- LMCache；
- 普通非 Beam 请求；
- Beam 和普通请求混合。

#### 交付物

```text
tests/unit/beam/
tests/contract/
tests/compatibility/
tests/integration/gpu/
tests/integration/npu/
```

#### 依赖

可以在 T0 后先建立测试目录和当前行为测试。

具体 Adapter 测试依赖对应 Adapter 完成。

#### 是否可并行

可以持续并行推进。

建议由独立测试负责人统筹，避免测试只验证实现细节。

---

### T12：建立 Ruff、mypy 和 CI 基线

#### 目标

使质量配置真正覆盖仓库主体代码，并逐步对齐上游。

#### 工作内容

- Ruff 启用明确规则：
  - `E`
  - `F`
  - `UP`
  - `B`
  - `SIM`
  - `I`
  - `G`
- 修正 pre-commit 重复 hook；
- 验证型 hook 不再默认 `--fix`；
- mypy 递归覆盖：
  - `vllm_gr/beam`
  - `vllm_gr/protocol`
  - `vllm_gr/config`
  - `vllm_gr/platform`
  - `vllm_gr/runtime`
- 对旧 patch 代码允许临时例外；
- 新模块禁止新增大范围 `Any`；
- 建立文件复杂度和文件行数报告；
- 统一标准测试命令。

#### 推进建议

不要一次性要求整个旧仓库通过 strict mypy。

建议采用增量规则：

```text
新模块：strict
已迁移模块：strict
旧 patch 模块：保留临时例外
完成迁移后：删除例外
```

#### 交付物

```text
pyproject.toml
.pre-commit-config.yaml
tools/quality/
CI workflow
```

#### 依赖

依赖 T0。

#### 是否可并行

可独立推进，但严格规则应分阶段启用，避免阻塞核心重构。

---

## 8. 第六阶段：目录结构重组

### T13：按照职责重新组织目录

#### 目标

让目录表达 GR 自身的领域和架构，而不是表达 patch 了哪些 vLLM 文件。

#### 前置条件

必须满足：

- T5 在线和离线状态机已基本统一；
- T8 Runner Adapter 已建立；
- T10 平台边界已建立；
- T2 兼容补丁已有统一入口。

#### 建议目录

```text
vllm_gr/
├── beam/
│   ├── domain.py
│   ├── state_machine.py
│   ├── coordinator.py
│   ├── ranking.py
│   └── pruning.py
│
├── catalog/
│   ├── models.py
│   ├── loader.py
│   └── constraints.py
│
├── config/
│   ├── models.py
│   ├── parser.py
│   └── validation.py
│
├── protocol/
│   ├── messages.py
│   ├── codec.py
│   └── version.py
│
├── adapters/
│   ├── offline/
│   ├── openai/
│   └── scheduler/
│
├── runtime/
│   ├── engine/
│   ├── model_runner/
│   └── attention/
│
├── platform/
│   ├── base.py
│   ├── gpu.py
│   └── npu.py
│
├── compat/
│   ├── patch_registry.py
│   ├── vllm/
│   ├── vllm_ascend/
│   └── lmcache/
│
├── metrics/
└── entrypoints/
```

#### 工作内容

- 移动已完成边界重构的模块；
- 删除空目录；
- 删除无意义的 `utils`；
- 将 `3rd_party` 改成合法 Python 包名；
- 修正 import；
- 保留必要的临时兼容 re-export；
- 避免一次 PR 同时进行大量逻辑修改和目录移动。

#### 依赖

依赖 T5、T8、T10。

#### 是否可并行

不建议多人同时大规模移动目录。

该任务适合由一名负责人统一执行，其他同事减少同时修改文件。

---

## 9. 第七阶段：旧实现清理

### T14：删除旧补丁、动态字段和重复实现

#### 目标

在新路径稳定后，删除旧架构，不长期维护两套实现。

#### 删除内容

- 全局 SamplingParams patch；
- EngineCoreRequestType Enum 内部修改；
- 完整 `process_input_sockets` 替换；
- Request 动态 Beam 字段；
- `SchedulerOutput.beam_data` 动态字段；
- 在线和离线重复 Beam 状态机；
- 根据 Runner 类名分支；
- `InputBatchProxy` 隐式状态修改；
- 顶层混合式 `patch.py`；
- 旧的 `engine_core_patch.py` 大文件；
- 失效的空目录；
- 重复 pre-commit 和测试脚本。

#### 依赖

依赖 T13，以及完整回归测试通过。

#### 是否可并行

可以按旧模块拆分删除，但应由一名负责人统一检查兼容入口是否仍被引用。

---

### T15：文档和最终质量收口

#### 目标

形成长期维护所需的开发文档。

#### 文档内容

```text
docs/architecture/overview.md
docs/architecture/beam_search.md
docs/architecture/engine_protocol.md
docs/architecture/platform_adapters.md
docs/compatibility/version_matrix.md
docs/development/testing.md
docs/development/adding_platform_support.md
```

同时更新：

- README；
- 安装依赖；
- 标准测试命令；
- 环境要求；
- vLLM 版本检查；
- vLLM-Ascend 版本检查；
- Beam Search 使用说明；
- 故障诊断方式。

#### 依赖

依赖主要架构任务完成。

#### 是否可并行

文档框架可以提前建立，最终内容在重构完成后统一更新。

---

## 10. 可并行推进的工作流

### 并行组 A：安全与工程基线

可同时推进：

```text
T1 SamplingParams 副作用修复
T2 补丁注册系统
T3 类型化配置
T12 Ruff/mypy/CI
```

这些任务之间逻辑依赖较少，但 T1 和 T2 可能同时修改 `patch.py` 和 `engine_core_patch.py`，需要提前划分文件区域。

---

### 并行组 B：领域与协议

可同时推进：

```text
T4 Beam 领域模型
T6 EngineCore 协议
T9 Attention 兼容层
```

三者分别关注：

- Beam 算法；
- Engine 消息；
- Attention 平台兼容。

文件重叠较少，适合不同同事并行处理。

---

### 并行组 C：Adapter 实现

接口确定后，可以并行：

```text
Offline Beam Adapter
OpenAI Beam Adapter
GPU ModelRunner Adapter
NPU ModelRunner Adapter
```

需要先共同评审接口，避免各自定义不同的数据模型。

---

### 并行组 D：测试

测试可以贯穿整个重构过程：

```text
领域单测
兼容契约测试
GPU 集成测试
NPU 集成测试
普通请求回归测试
```

测试负责人不应只跟随实现代码编写 Stub，而应独立维护行为契约。

---

## 11. 必须串行推进的关键链路

### 关键链路一：Beam 领域重构

```text
T4 Beam 领域模型
    ↓
T5 统一在线和离线状态机
    ↓
T13 目录调整
    ↓
T14 删除旧实现
```

---

### 关键链路二：Engine 扩展重构

```text
T6 EngineCore 协议
    ↓
T7 Scheduler Beam Context
    ↓
T8 ModelRunner Adapter
    ↓
T10 GPU/NPU 平台策略
    ↓
T13 目录调整
```

---

### 关键链路三：补丁治理

```text
T2 补丁注册系统
    ↓
T9 Attention 兼容补丁迁移
    ↓
其他 vLLM/vLLM-Ascend 补丁迁移
    ↓
T14 删除顶层混合 patch.py
```

---

## 12. 文件冲突和负责人建议

### 高冲突文件

以下文件不适合多人同时修改：

```text
vllm_gr/patch.py
vllm_gr/v1/engine/engine_core_patch.py
vllm_gr/entrypoints/gr.py
vllm_gr/entrypoints/openai/serving_engine.py
vllm_gr/v1/attention/backends/beam_attn.py
```

### 建议所有权

| 文件或区域 | 建议所有权 |
|---|---|
| `patch.py` | 补丁框架负责人 |
| `engine_core_patch.py` | Engine/协议负责人 |
| `entrypoints/gr.py` | Offline Adapter 负责人 |
| `entrypoints/openai/serving_engine.py` | OpenAI Adapter 负责人 |
| `beam_attn.py` | Attention/平台负责人 |

### 原则

同一个高冲突文件在一个阶段内尽量只有一名主要负责人。

其他同事通过新增模块完成重构，不直接同时修改旧文件。最后由文件负责人完成旧逻辑替换。

---

## 13. 推荐团队拆分

### 角色 A：架构与兼容负责人

负责：

```text
T0
T2
T13
T14
```

关注整体依赖方向、兼容边界和最终收口。

---

### 角色 B：Beam 领域负责人

负责：

```text
T4
T5 中的 Coordinator
Beam 领域单元测试
```

要求熟悉 Beam Search 算法，但不需要深入 NPU。

---

### 角色 C：Engine 和 Scheduler 负责人

负责：

```text
T6
T7
EngineCore 契约测试
```

要求熟悉：

- EngineCoreRequest；
- EngineCoreClient；
- ZMQ；
- Scheduler；
- KV Cache 生命周期。

---

### 角色 D：GPU Runtime 负责人

负责：

```text
T8 GPU Adapter
GPU Attention 路径
GPU 集成测试
```

---

### 角色 E：NPU 和 vLLM-Ascend 负责人

负责：

```text
T9
T8 NPU Adapter
T10 NPU Platform
NPU 集成测试
```

---

### 角色 F：OpenAI 和 Offline Adapter 负责人

可以进一步拆成两人：

```text
Offline Adapter
OpenAI Adapter
```

共同基于 T5 的 Coordinator 工作。

---

### 角色 G：工程质量负责人

负责：

```text
T11
T12
T15
```

该角色应独立维护测试层级、CI 和兼容矩阵。

---

## 14. PR 拆分原则

每个 PR 应尽量只做一种类型的变化。

推荐：

```text
PR 1：增加新接口和新模块
PR 2：为旧实现增加 Adapter
PR 3：迁移一个运行路径
PR 4：增加兼容测试
PR 5：删除旧实现
PR 6：移动目录和修复 import
```

不推荐在同一个 PR 中同时进行：

- 大规模文件移动；
- 核心算法修改；
- EngineCore 协议修改；
- NPU 行为修改；
- 测试基线修改。

否则 Review 很难区分：

```text
代码只是移动了
代码被重构了
代码行为发生了改变
```

---

## 15. 每个任务的完成标准

每个子任务应至少满足以下条件：

1. 有明确输入和输出；
2. 有单元测试或契约测试；
3. 不增加新的全局 monkey patch；
4. 不增加新的动态属性；
5. 不直接在新代码中扩散 vLLM 私有 API；
6. 私有 API 必须限制在 `compat` 或 Adapter 内；
7. 普通非 Beam 请求行为不变；
8. GPU/NPU 差异不进入 Beam 领域层；
9. 新模块通过 Ruff 和 mypy；
10. 文档说明该模块的职责和依赖方向。

---

## 16. 推荐执行顺序

### 第一批

```text
T0 现状基线
T1 SamplingParams 安全修复
T2 补丁注册框架
T3 类型化配置
T4 Beam 领域模型
T6 EngineCore 协议设计
T9 Attention 兼容隔离
T12 工程工具基线
```

这批任务大部分可以并行。

### 第二批

```text
T5 统一 Beam 状态机
T7 Scheduler Beam Context
T11 契约测试体系
```

### 第三批

```text
T8 GPU/NPU ModelRunner Adapter
T10 GPU/NPU 平台策略
```

### 第四批

```text
T13 目录重组
T14 删除旧代码
T15 文档和质量收口
```

---

## 17. 最重要的执行约束

本次重构需要避免以下做法：

1. 不要先大规模移动目录；
2. 不要在拆分文件时复制现有逻辑；
3. 不要同时长期保留两套 Beam 状态机；
4. 不要继续向上游对象增加动态字段；
5. 不要为每个新问题增加新的 monkey patch；
6. 不要让 GPU/NPU 判断继续散落；
7. 不要让新模块直接依赖 OpenAI Serving；
8. 不要一次性修改 Engine、Scheduler、Runner、Attention 和目录；
9. 不要只依靠 Stub 测试判断上游兼容；
10. 不要在没有兼容测试的情况下删除旧路径。

最终目标不是让补丁文件变小，而是让补丁只存在于明确、版本化、可删除的兼容边界中。
