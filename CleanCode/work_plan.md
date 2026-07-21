# vLLM-GR 两人重构分工方案

## 1. 适用场景

当前重构团队由两名开发者组成：

- **A：经验丰富同事**
- **B：经验较浅同事**

本方案的目标是：

1. 控制 EngineCore、Scheduler、ModelRunner 和 NPU 路径的改造风险；
2. 避免把项目拆成过多 Task 和 Sub-task；
3. 保证每个工作包只有一个明确责任人；
4. 让经验较浅同事能够逐步参与核心模块，而不是长期只负责测试和文档；
5. 降低 Issue、分支、Review 和依赖管理成本。

原有 T0–T15 继续保留，作为完整架构重构的检查清单。

实际执行时，不再为每个细节创建 Sxx 子任务，而是将工作合并为 7 个可独立交付的工作包：

```text
W1 ～ W7
```

---

## 2. 分工原则

### 2.1 一个工作包只有一个责任人

每个工作包必须明确：

```text
Owner
Reviewer
Dependencies
Deliverables
Acceptance Criteria
```

Owner 独立负责：

- 方案；
- 实现；
- 测试；
- 文档；
- 解决 Review 意见；
- 合并前验证。

Reviewer 只负责：

- 检查设计；
- 检查风险；
- 提出修改意见；
- 批准或拒绝合并。

不在同一个工作包中将实现任务平均分给两个人。

---

### 2.2 高风险底层改造由经验丰富同事负责

以下工作应主要由 A 负责：

- 全局 monkey patch；
- EngineCore 多进程协议；
- ZMQ 和 msgspec；
- Scheduler 和 KV Cache 生命周期；
- GPU/NPU ModelRunner 私有接口；
- vLLM-Ascend Attention 兼容；
- 删除旧核心路径的最终判断。

---

### 2.3 边界明确、容易验证的能力由经验较浅同事负责

以下工作适合由 B 独立负责：

- 当前行为测试基线；
- Ruff、mypy 和 CI；
- 类型明确的领域模型；
- Beam 排序和剪枝算法；
- Offline Adapter；
- 目录和 import 迁移；
- 回归测试；
- 文档。

---

### 2.4 不要一开始移动目录

目录迁移必须等待以下边界稳定：

- Beam 领域模型；
- EngineCore 和 Scheduler 协议；
- ModelRunner Adapter；
- GPU/NPU 平台边界；
- 兼容补丁入口。

否则只是在新目录中保留旧耦合关系。

---

## 3. 工作包总览

| 工作包 | 范围 | 责任人 | 主要依赖 |
|---|---|---|---|
| W1 | 当前行为基线与工程治理 | B | 无 |
| W2 | 全局补丁与配置治理 | A | W1 的关键测试 |
| W3 | Beam 领域模型与离线统一 | B | W1 |
| W4 | EngineCore 与 Scheduler 契约 | A | W2 |
| W5 | 在线 Beam Search 统一 | A | W3、W4 |
| W6 | ModelRunner、Attention 与平台隔离 | A | W4 |
| W7 | 目录迁移、旧代码清理与收口 | B | W3、W5、W6 |

---

## 4. 工作包依赖关系

```text
W1 当前行为基线与工程治理
│
├── W2 全局补丁与配置治理
│   └── W4 EngineCore 与 Scheduler 契约
│       ├── W5 在线 Beam Search 统一
│       └── W6 ModelRunner、Attention 与平台隔离
│
└── W3 Beam 领域模型与离线统一
    └── W5 在线 Beam Search 统一

W3 + W5 + W6
        ↓
W7 目录迁移、旧代码清理与收口
```

---

## 5. W1：当前行为基线与工程治理

### Owner

B：经验较浅同事

### Reviewer

A：经验丰富同事

### 对应原 Task

```text
T0 现状基线与兼容矩阵
T11 测试分层中的基线部分
T12 Ruff、mypy 和 CI 基线
```

### 目标

在修改核心架构前建立安全网，并修正当前工程工具失真问题。

### 工作内容

- 整理当前运行模式：
  - Offline；
  - OpenAI Serving；
  - GPU；
  - NPU；
  - 单 Engine；
  - 多 Engine；
  - LMCache；
  - Catalog；
  - Beam Attention。
- 整理现有 monkey patch 清单；
- 标记每个补丁生效的进程：
  - Frontend；
  - EngineCore；
  - Worker；
  - 所有进程。
- 建立兼容矩阵：
  - vLLM 0.22.1；
  - vLLM-Ascend 0.22.1rc；
  - torch；
  - torch-npu。
- 固定普通请求和 Beam 请求的当前行为；
- 增加 Offline/Online smoke test；
- 修正 Ruff 配置；
- 修正 mypy 覆盖范围；
- 清理重复 pre-commit hook；
- 统一测试入口；
- 修正文档中的测试命令。

### 建议交付物

```text
docs/refactoring/current_behavior.md
docs/refactoring/compatibility_matrix.md
docs/refactoring/patch_inventory.md
tests/compatibility/test_current_behavior.py
pyproject.toml
.pre-commit-config.yaml
tools/quality/
```

### 验收条件

- 普通非 Beam 请求有回归测试；
- Offline Beam 有回归测试；
- Online Beam 有基本 smoke test；
- Ruff 和 mypy 命令可以稳定执行；
- 标准测试入口唯一且文档一致；
- 后续核心改造可以通过测试判断是否改变现有行为。

---

## 6. W2：全局补丁与配置治理

### Owner

A：经验丰富同事

### Reviewer

B：经验较浅同事

### 对应原 Task

```text
T1 修复 SamplingParams 全局副作用
T2 补丁注册与原子初始化
T3 类型化 GR 配置
```

### 目标

收敛全局副作用，建立可靠、可验证的补丁生命周期和统一配置入口。

### 工作内容

#### SamplingParams

- 删除或限制：

```python
SamplingParams._verify_greedy_sampling = lambda self: None
```

- GR 特殊参数只在 GR 请求入口转换；
- 普通请求继续使用上游校验逻辑。

#### Patch Registry

建立统一补丁定义：

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

补丁状态至少包括：

```text
NOT_APPLIED
APPLIED
SKIPPED
FAILED
INCOMPATIBLE
```

必需补丁失败时应阻止服务继续启动。

#### 类型化配置

建立统一配置模型：

```python
@dataclass
class GRConfig:
    enabled: bool
    catalog_path: str | None
    attention_backend: str | None
    beam: BeamConfig
    compatibility: CompatibilityConfig
```

禁止其他模块继续直接读取：

```python
additional_config["vllm_gr_*"]
```

### 建议交付物

```text
vllm_gr/config/models.py
vllm_gr/config/parser.py
vllm_gr/config/validation.py
vllm_gr/compat/patch_registry.py
vllm_gr/compat/patch_result.py
vllm_gr/compat/version_check.py
```

### 验收条件

- 普通请求不再受 GR 参数补丁影响；
- 必需补丁失败时启动失败；
- 所有补丁有明确状态和日志；
- 配置只有一个解析入口；
- 配置错误在启动阶段发现；
- 不新增新的顶层全局 monkey patch。

---

## 7. W3：Beam 领域模型与离线统一

### Owner

B：经验较浅同事

### Reviewer

A：经验丰富同事

### 对应原 Task

```text
T4 Beam Search 领域模型
T5 中的 Offline 迁移部分
```

### 目标

抽离不依赖 vLLM、OpenAI、GPU/NPU 的 Beam Search 领域逻辑，并先统一离线路径。

### 工作内容

建立领域对象：

```text
BeamState
BeamCandidate
BeamGroup
BeamStepInput
BeamStepResult
BeamFinishedReason
BeamRequestContext
```

抽取纯领域操作：

- Candidate 扩展；
- Candidate 排序；
- Beam prune；
- EOS 判断；
- Length penalty；
- 父子 Beam 映射；
- Segment 去重；
- Finished Beam 管理；
- 最终结果排序。

建立 Offline Adapter：

```text
OfflineBeamEngineAdapter
```

将 `entrypoints/gr.py` 中的 Beam Search 逻辑迁移到共享领域层。

### 建议交付物

```text
vllm_gr/beam/domain.py
vllm_gr/beam/ranking.py
vllm_gr/beam/pruning.py
vllm_gr/beam/state_machine.py
vllm_gr/adapters/offline/beam_engine.py
tests/unit/beam/
```

### 约束

领域层不得依赖：

- AsyncLLM；
- OpenAI 协议；
- Scheduler；
- ModelRunner；
- GPU/NPU；
- ZMQ；
- vLLM 私有类型。

### 验收条件

- Beam 排序、剪枝、EOS、Length Penalty 有纯单元测试；
- Offline Beam 使用新的领域模型；
- 离线路径不再维护完整重复 Beam 算法；
- 输出与 W1 基线一致；
- 新领域模块通过 strict mypy。

---

## 8. W4：EngineCore 与 Scheduler 契约

### Owner

A：经验丰富同事

### Reviewer

B：经验较浅同事

### 对应原 Task

```text
T6 EngineCore 扩展协议
T7 Scheduler Beam 上下文
```

### 目标

删除运行时修改 Enum 和动态字段传递，建立显式、可版本化的 EngineCore 和 Scheduler 契约。

### 工作内容

定义消息对象：

```text
GRAddBatchRequest
GRBeamStepUpdateRequest
GRRequestEnvelope
GRResponseEnvelope
GRProtocolVersion
```

评估：

- 复用上游 Utility Request；
- 或建立独立 GR Envelope。

建立请求和调度对象：

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
    ...
```

清理：

- EngineCoreRequestType Enum 内部修改；
- Request 动态 Beam 字段；
- `SchedulerOutput.beam_data`；
- 完整 `process_input_sockets` 覆盖。

### 建议交付物

```text
vllm_gr/protocol/messages.py
vllm_gr/protocol/version.py
vllm_gr/protocol/codec.py
vllm_gr/scheduler/request_metadata.py
vllm_gr/scheduler/context.py
vllm_gr/scheduler/adapter.py
tests/contract/engine_protocol/
```

### 验收条件

- 不再修改 `_member_map_`、`_value2member_map_`；
- Client 和 EngineCore 有协议版本检查；
- 未知消息有明确失败行为；
- Scheduler 和 Runner 间使用显式上下文；
- 不再动态向上游对象增加 Beam 字段；
- 尽量删除 `process_input_sockets` 整体替换。

---

## 9. W5：在线 Beam Search 统一

### Owner

A：经验丰富同事

### Reviewer

B：经验较浅同事

### 对应原 Task

```text
T5 中的共享 Coordinator 和 OpenAI 迁移部分
```

### 目标

让 OpenAI/Async 和 Offline 使用同一 Beam 状态机。

### 工作内容

实现：

```text
BeamSearchCoordinator
AsyncBeamEngineAdapter
```

OpenAI 入口只保留：

- 请求协议转换；
- Engine 调用；
- 流式输出；
- 异常转换；
- abort；
- 清理；
- 最终 OpenAI 响应转换。

删除 OpenAI Serving 中重复维护的：

- Candidate 排序；
- Beam prune；
- EOS；
- Length Penalty；
- Finished Beam；
- Segment 去重；
- 父子映射。

### 建议交付物

```text
vllm_gr/beam/coordinator.py
vllm_gr/adapters/openai/beam_engine.py
```

### 验收条件

- Online 和 Offline 使用同一个领域状态机；
- Online/Offline 相同输入的 Beam 结果保持一致；
- Streaming、abort、异常和资源清理测试通过；
- OpenAI Serving 不再承载完整 Beam 算法。

---

## 10. W6：ModelRunner、Attention 与平台隔离

### Owner

A：经验丰富同事

### Reviewer

B：经验较浅同事

### 对应原 Task

```text
T8 ModelRunner Adapter
T9 Attention Backend 兼容隔离
T10 GPU/NPU 平台策略
```

### 目标

把 vLLM/vLLM-Ascend 私有 API、GPU/NPU 差异和 Attention 兼容集中到明确边界。

### 工作内容

定义 Runner Adapter：

```python
class BeamModelRunnerAdapter(Protocol):
    def prepare_beam_inputs(...): ...
    def update_after_execution(...): ...
    def collapse_beam_outputs(...): ...
```

实现：

```text
GPUBeamModelRunnerAdapter
NPUBeamModelRunnerAdapter
```

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

隔离 vLLM-Ascend CUSTOM Attention Backend 兼容逻辑：

```text
vllm_gr/compat/vllm_ascend/v0_22_1rc/
```

清理：

- 根据类名字符串判断 GPU/NPU；
- InputBatchProxy 属性读取产生副作用；
- 私有 ModelRunner API 在多个模块中扩散；
- Attention Selector 私有补丁散落在顶层。

### 建议交付物

```text
vllm_gr/runtime/model_runner/base.py
vllm_gr/runtime/model_runner/gpu.py
vllm_gr/runtime/model_runner/npu.py
vllm_gr/platform/base.py
vllm_gr/platform/gpu.py
vllm_gr/platform/npu.py
vllm_gr/compat/vllm_ascend/v0_22_1rc/attention_backend.py
tests/contract/model_runner/
tests/compatibility/test_ascend_attention_backend.py
```

### 验收条件

- GPU/NPU 差异只存在于平台和 Adapter 层；
- 不根据 Runner 类名字符串分支；
- vLLM 私有 API 只存在于 `compat` 或 Adapter 内；
- Tensor 状态修改有明确生命周期和恢复机制；
- GPU 和 NPU 契约测试通过；
- Attention Backend 兼容补丁有版本检查。

---

## 11. W7：目录迁移、旧代码清理与收口

### Owner

B：经验较浅同事

### Reviewer

A：经验丰富同事

### 对应原 Task

```text
T13 目录结构重组
T14 删除旧补丁和重复实现
T15 文档、迁移和最终收口
```

### 目标

在新边界稳定后完成目录迁移、旧实现删除、测试和文档收口。

### 工作内容

迁移为职责导向目录：

```text
vllm_gr/
├── beam/
├── catalog/
├── config/
├── protocol/
├── adapters/
├── runtime/
├── platform/
├── compat/
├── metrics/
└── entrypoints/
```

清理：

- 空目录；
- 无意义 `utils`；
- `3rd_party` 非法包名；
- 重复 Beam Search；
- 动态字段；
- 旧 Patch；
- 旧 God Module；
- 重复测试脚本；
- 失效 import；
- 过渡 re-export。

更新：

- README；
- 架构文档；
- 兼容矩阵；
- 测试说明；
- 平台开发说明；
- 迁移说明。

### 建议交付物

```text
docs/architecture/overview.md
docs/architecture/beam_search.md
docs/architecture/engine_protocol.md
docs/architecture/platform_adapters.md
docs/compatibility/version_matrix.md
docs/development/testing.md
docs/development/adding_platform_support.md
```

### 验收条件

- 新目录能表达真实职责；
- 不保留长期双实现；
- 普通请求、Offline Beam、Online Beam 回归通过；
- GPU/NPU 兼容测试通过；
- 旧补丁和旧动态字段已删除；
- 文档和代码结构一致；
- A 完成最终架构验收。

---

## 12. 推荐执行节奏

### 第一轮

A：

```text
W2 全局补丁与配置治理
```

B：

```text
W1 当前行为基线与工程治理
```

W2 可以在 W1 的关键回归测试落地后开始，不必等待 W1 所有文档完成。

---

### 第二轮

A：

```text
W4 EngineCore 与 Scheduler 契约
```

B：

```text
W3 Beam 领域模型与离线统一
```

这两个工作包文件重叠较少，适合并行。

---

### 第三轮

A：

```text
W5 在线 Beam Search 统一
```

B：

- 完成 W3 回归；
- 补充 W5 所需的行为测试；
- 维护兼容矩阵。

W5 完成后，A 继续：

```text
W6 ModelRunner、Attention 与平台隔离
```

B 在此期间补充：

- Runner 契约测试；
- GPU/NPU 回归场景；
- 文档框架。

这些仍属于对应工作包的支持工作，不单独建立新的项目任务。

---

### 第四轮

B：

```text
W7 目录迁移、旧代码清理与收口
```

A：

- Review 高风险删除；
- 验证 EngineCore、Scheduler、NPU 和 Attention 路径；
- 完成最终架构验收。

---

## 13. 文件所有权建议

### A 主要拥有

```text
vllm_gr/patch.py
vllm_gr/v1/engine/engine_core_patch.py
vllm_gr/v1/engine/core.py
vllm_gr/v1/engine/core_client.py
vllm_gr/v1/attention/backends/beam_attn.py
vllm_gr/entrypoints/openai/serving_engine.py
```

### B 主要拥有

```text
vllm_gr/config/
vllm_gr/beam/
vllm_gr/adapters/offline/
tests/unit/
tests/compatibility/
tools/quality/
docs/
```

同一个高冲突文件在一个阶段内尽量只有一个主要修改者。

---

## 14. 每个工作包的 Issue 模板

```md
# Wn 工作包名称

Owner:
Reviewer:
Dependencies:

## Goal

## Scope

## Out of Scope

## Checklist

- [ ] ...
- [ ] ...
- [ ] ...

## Deliverables

## Acceptance Criteria

## Affected Files

## Risks
```

---

## 15. PR 拆分建议

每个工作包允许包含 1～3 个 PR，但不必把每个 PR 都建立为独立项目任务。

推荐：

```text
PR 1：增加新接口和新模块
PR 2：迁移旧运行路径
PR 3：删除旧实现并完成回归
```

不建议在一个 PR 中同时进行：

- 大规模目录移动；
- EngineCore 协议修改；
- 核心 Beam 算法修改；
- NPU ModelRunner 修改；
- 删除全部旧代码；
- 修改 CI 基线。

---

## 16. 工作量与责任比例

代码量可以接近：

```text
A：55%
B：45%
```

但架构决策和风险责任建议为：

```text
A：80%
B：20%
```

A 不一定写更多代码，但应负责：

- 核心接口；
- 高风险运行时；
- 上游兼容；
- PR 合并顺序；
- 最终旧路径删除判断。

B 负责较多边界清晰的新模块、测试、目录和文档，并逐步深入 Beam 和 Adapter 代码。

---

## 17. 经验较浅同事的成长路径

建议 B 按以下顺序深入代码：

```text
测试基线
    ↓
类型化配置与质量工具
    ↓
Beam 纯领域逻辑
    ↓
Offline Adapter
    ↓
协议和 Runner 契约测试
    ↓
目录迁移与系统回归
```

不建议一开始独立承担：

```text
EngineCore 协议
process_input_sockets
NPU ModelRunner
ACL Graph
全局 Patch 初始化
Scheduler / KV Cache 生命周期
```

---

## 18. 最终建议

对于两人团队，适合的管理粒度是：

```text
7 个工作包
每个工作包 1 个 Owner
每个工作包 1 个 Reviewer
每个工作包 5～10 个 Checklist
每个工作包 1～3 个 PR
```

原 T0–T15 继续作为完整重构检查清单。

实际开发、分工和进度管理统一使用 W1–W7，避免维护过多 Sxx 子任务和跨 Issue 依赖。
