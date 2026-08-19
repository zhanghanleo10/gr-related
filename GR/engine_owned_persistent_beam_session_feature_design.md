# vLLM-GR Persistent Batch Beam Session 与 Worker-Owned Beam Runtime 总体方案

> 状态：Feature Design  
> 日期：2026-08-19  
> 目标仓库：JiusiServe/vllm-gr  
> 背景 RFC：[Serving Beam Controller 下沉与 Persistent Beam Session](https://github.com/zhanghanleo10/vllm-gr/issues/35)  
> 当前实现参考：[PR #290](https://github.com/JiusiServe/vllm-gr/pull/290)  
> 文档范围：总体架构、核心抽象、资源所有权和长期演进方向

---

## 0. 一页结论

这项特性不是简单地把 Serving 中的循环搬进 EngineCore，也不是把现有
BeamSearchController 原样复制到 Worker。目标是重新建立三条稳定边界：

1. Online Serving 与 Offline API 只是平级入口。
2. EngineCore 和 Scheduler 负责 Session、Request、调度与资源生命周期。
3. Worker 负责约束过滤、Beam 选择、Beam 状态和 Beam KV 的完整数据闭环。

业务侧一次提交的 BatchRequest 包含 B 个输入。它在系统内表示为一个
BatchBeamSession 和 B 个原生 child Requests，而不是一个包含 B 条并行序列的
原生 vLLM Request。

~~~mermaid
flowchart TB
    subgraph FRONTENDS["Peer Frontends"]
        direction LR
        ONLINE["Online Serving"]
        OFFLINE["Offline API"]
    end

    API["Unified Engine Beam API"]
    COORD["BatchBeamCoordinator"]
    CHILD["B Native Child Requests"]
    SCHED["vLLM Scheduler"]
    PREFILL["Native Prefill + Paged KV + Prefix Cache"]
    BARRIER["Prefill Barrier"]
    RUNTIME["Worker BeamRuntime"]
    FINAL["One BatchBeamFinalResult"]

    ONLINE --> API
    OFFLINE --> API
    API --> COORD
    COORD --> CHILD
    CHILD --> SCHED
    SCHED --> PREFILL
    PREFILL --> BARRIER
    BARRIER --> RUNTIME
    RUNTIME --> SCHED
    SCHED --> RUNTIME
    RUNTIME --> FINAL
    FINAL --> ONLINE
    FINAL --> OFFLINE
~~~

约束资源采用模型级加载方式：

~~~text
部署配置
→ Worker.load_model
→ 加载模型权重
→ 加载一个默认 Constraint Artifact
→ 解析为常驻 Host 或 Device Buffer
→ 显存探测与 KV 分配
→ Warmup
~~~

当前只支持一个默认约束资源。内部接口保留 resource_id，但不支持一个模型实例
运行期加载多张表，也不支持按请求发送或替换完整约束资源。

---

## 1. 目的与问题定义

### 1.1 当前问题

传统链路把一个连续 Beam Session 拆成了多个跨层往返：

~~~text
Serving 计算 Beam decision
→ EngineCore / Scheduler 执行一步
→ Worker 返回候选
→ EngineCore 或 Serving 再计算下一步
~~~

这会导致：

- Beam 算法状态与 Worker Beam KV 分属不同所有者；
- W × K 候选、scores 和 parent 信息跨进程传输；
- EngineCore 需要理解 CPU Trie、候选过滤和 Beam materialization；
- Online 与 Offline 容易形成不同控制路径；
- 后续替换 Device Constraint Table 时还要再次移动控制边界；
- Prefill Prefix KV、Beam suffix KV 和 Session 终态难以形成同一个生命周期。

### 1.2 目标结果

即使移除 Online Serving，Engine 仍应能够独立完成一个业务 BatchRequest：

~~~text
提交一次
→ B 个 child Requests 使用原生 Prefill
→ Worker 完成全部 Beam Decode
→ 返回一次 Batch Final Result
~~~

这项特性的核心目标是所有权正确，而不是某一个算子或 IPC 的局部优化。

---

## 2. 已冻结的设计不变量

1. 一个业务 BatchRequest 对应一个稳定的 BatchBeamSession。
2. BatchBeamSession 包含 B 个 item，并映射为 B 个原生 child Requests。
3. B 个 child Requests 分别使用原生 Scheduler、Paged KV 和 Prefix Cache。
4. Prefill 可以独立调度、分块和复用，但进入 Beam Decode 前需要 Batch barrier。
5. Worker 是 Beam scores、sequence、constraint state、parent 和 Beam KV 的唯一权威。
6. Constraint filter、candidate select、global Beam select 和 final select 全部位于 Worker。
7. EngineCore 只保存调度与生命周期所需的协调状态，不保存完整 Beam 算法状态。
8. 每一次 Prefill 或 Decode execution 仍然由 Scheduler 产生，EngineCore 不直接绕过
   Scheduler 调用 Worker。
9. 一个模型实例启动时只加载一个默认约束资源。
10. 约束资源生命周期与模型一致，当前不涉及热更新、逐出和用户关联 KV。
11. 当前执行 backend 使用 CPU Trie，后续可以替换成 Device Constraint Table。
12. Online Serving 与 Offline API 共用相同 Engine、Scheduler 和 Worker Runtime。
13. Frontend 只接收最终结果，不接收中间 Prefill 或 Decode 候选。
14. 初始实现可以限制一个 Session 只有一个 in-flight epoch，但不能把该限制写死进
    Session 数据模型。

### 2.1 当前非目标

- 用户级、租户级或会话级约束资源动态加载；
- 一个模型实例同时维护多个可选 constraint resource；
- 约束资源热更新和版本并存；
- Prefill 用户关联 KV；
- 将本方案限定为某一种 Full Graph 实现；
- 在本文中展开具体子任务、Review、回滚、性能或发布验收。

---

## 3. 总体架构与责任边界

~~~mermaid
flowchart TB
    subgraph FRONTEND["① Serving / API Layer"]
        ON["Online Adapter"]
        OFF["Offline Adapter"]
    end

    subgraph ENGINE["② EngineCore"]
        direction TB
        API["EngineBeamAPI"]
        COORD["BatchBeamCoordinator"]
        ROUTER["FinalResultRouter"]

        subgraph SCHEDULER["Scheduler"]
            direction LR
            REQS["B Persistent Child Requests"]
            PREFIX["Native Prefix KV Ownership"]
            BUDGET["Token and Capacity Admission"]

            REQS --> PREFIX
            REQS --> BUDGET
        end

        API --> COORD
        COORD --> REQS
        COORD --> ROUTER
    end

    subgraph WORKER["③ Worker / ModelRunner"]
        direction TB
        LOAD["ConstraintHeadConfig + Default Artifact"]
        CRM["ConstraintResourceManager"]
        CHE["ConstraintHeadExtension"]
        SESSION["WorkerBatchBeamSession"]
        SELECT["BeamSelector"]
        BKV["BeamKVManager"]

        LOAD --> CRM
        CRM --> CHE
        SESSION --> CHE
        CHE --> SELECT
        SELECT --> BKV
    end

    ON --> API
    OFF --> API
    REQS --> SESSION
~~~

主图只表达自顶向下的所有权与下行执行链：

~~~text
Serving / Offline
→ EngineCore
→ Scheduler
→ Worker / ModelRunner
~~~

WorkerBeamStepResult 沿相同边界返回 BatchBeamCoordinator；最终结果由
FinalResultRouter 返回对应 Frontend。返回路径没有再画入主图，避免反向箭头破坏层次关系。

### 3.1 责任矩阵

| 组件 | 核心职责 | 不应负责 |
| --- | --- | --- |
| Online Adapter | 参数校验、tokenize、cancel、协议格式化 | Beam 主循环、Trie、KV |
| Offline Adapter | Batch 提交、job 适配、结果组织 | 独立调度器、独立 Beam 算法 |
| EngineBeamAPI | 请求规范化、Session 创建、最终 future | 文件解析、设备算子 |
| BatchBeamCoordinator | B 个 child 映射、barrier、epoch、终态协调 | 候选过滤、Beam score、KV reorder |
| Scheduler | child Request、Prefill、budget、Prefix KV 生命周期 | 完整 Beam sequence 和 constraint state |
| ConstraintResourceManager | 默认资源加载、校验和常驻 Buffer | 每请求动态资源注册 |
| ConstraintHeadExtension | 根据约束资源生成合法候选 | Session 生命周期、Frontend 输出 |
| BeamSelector | 累积分数、global top-W、parent 选择 | Paged KV block 分配 |
| WorkerBatchBeamSession | B × W 状态、step、mask、终态 | Scheduler queue |
| BeamKVManager | suffix KV 写入、选择和释放 | Prompt tokenize |
| FinalResultRouter | 一次性路由内部终态 | 下一步 Beam decision |

---

## 4. 业务 Batch、Session 与原生 Request

### 4.1 三个不同概念

~~~text
业务 BatchRequest
    ↓ 一次外部调用
BatchBeamSession
    ↓ 组织 B 个 item 的生命周期
B Native Child Requests
    ↓ 分别进入 vLLM Scheduler
Physical Execution Batch
    ↓ 每个 tick 动态组装的实际执行行
~~~

三者不能合并：

- BatchBeamSession 是业务和算法生命周期单位；
- child Request 是 Scheduler 和 Prefix Paged KV 的管理单位；
- physical batch 是某个调度 tick 的执行单位。

### 4.2 为什么不能把 B 合成一个原生 Request

原生 Request 表达一条串行 token history，而业务 Batch 中的 B 个输入是 B 条独立序列。
如果把它们拼成一个 Request，会破坏：

- 每个 prompt 的 position 和长度；
- Prefix Cache hash 与 block 命中；
- chunked prefill 进度；
- KV block 所有权；
- 单 item 错误与终态；
- Scheduler 的 token budget。

因此采用稳定映射：

~~~python
@dataclass
class BatchBeamCoordinator:
    batch_session_id: str
    child_request_ids: tuple[str, ...]  # 长度 B
    prefill_ready_mask: tuple[bool, ...]
    stage: str
    next_epoch: int
    in_flight_epoch: int | None
    worker_session_handle: str | None
~~~

### 4.3 一个 Session 内允许多个物理 batch

允许。

当 B × W 超过一个 execution tick 的预算时，同一个 Session 的一个逻辑 step 可以被拆成
多个物理 micro-batches：

~~~mermaid
flowchart LR
    STEP["Logical Step e"]
    M1["Micro-batch 1"]
    M2["Micro-batch 2"]
    M3["Micro-batch N"]
    ACC["Worker StepAccumulator"]
    SEL["Per-item Global Beam Select"]
    COMMIT["Atomic Epoch Commit"]

    STEP --> M1
    STEP --> M2
    STEP --> M3
    M1 --> ACC
    M2 --> ACC
    M3 --> ACC
    ACC --> SEL
    SEL --> COMMIT
~~~

但必须满足：

- 一个 item 的 W 个 parent lanes 收集完整后，才能进行该 item 的 global top-W；
- B 个 item 不互相竞争 Beam，global top-W 在每个 item 内独立执行；
- 一个 epoch 的状态不能在只收到部分 micro-batch 时提交；
- row 顺序不能作为身份，必须使用 session、epoch、item 和 beam lane 映射。

多个独立 Session 未来也可以组成同一个物理 execution batch，但它们仍保持独立的
Coordinator、Worker Session、Prefix KV、epoch 和终态。

---

## 5. Prefill：完整复用原生 vLLM

### 5.1 每个 child Request 独立 Prefill

~~~mermaid
sequenceDiagram
    participant F as Frontend
    participant E as EngineCore
    participant S as Scheduler
    participant W as Worker
    participant R as Worker BeamRuntime

    F->>E: submit BatchRequest with B items
    E->>S: admit B child Requests
    loop Native continuous scheduling
        S->>S: prefix lookup and token budget
        S->>W: schedule child Prefill chunks
        W-->>S: native Prefill progress
        S-->>E: child Prefill ready
    end
    E->>E: prefill_ready_mask all true
    E->>R: create BatchBeamSession
    R-->>E: WorkerSessionHandle
    E->>S: enter Beam Decode
~~~

每个 child Request 继续使用原生能力：

- get_computed_blocks 与 Prefix Cache 命中；
- Paged KV block 分配和引用计数；
- chunked prefill；
- continuous batching；
- Scheduler token budget；
- 不同 prompt 长度独立推进。

本方案不建立用户关联 KV，也不改变原生 Prefix Cache 的匹配和逐出策略。

### 5.2 Prefill Barrier

早完成的 child 可以先保存 Prefill logits 和 Prefix KV，但整个 BatchBeamSession 只有在
prefill_ready_mask[B] 全部为真后才能进入共同 Beam Decode。

~~~text
child 0 ready ─┐
child 1 ready ─┼─→ Batch Prefill Barrier → initialize B × W Beam state
...            │
child B-1 ready┘
~~~

Worker 可以提前执行单个 item 的初始 constrained selection，但 Batch Session 的 Decode
不能在其他 item 尚未完成 Prefill 时开始。

### 5.3 Prefix KV 生命周期

每个 child Request 的 Prefix blocks 在 Session 期间保持绑定：

~~~text
Native Prefill
→ Prefix Paged KV retained
→ Beam Decode reads shared Prefix KV
→ Beam suffix written into dedicated Beam KV
→ Worker teardown acknowledged
→ child Requests and Prefix bindings released
~~~

Scheduler 继续是 Prefix KV 的唯一所有者。Worker 不复制一套 Prompt Paged KV，也不自行
释放原生 block。

---

## 6. Worker-Owned Beam Runtime

### 6.1 为什么整个后处理闭环都要下沉

约束过滤、Beam select 和 KV reorder 共享同一组数据：

- logits 和 per-parent candidates；
- cumulative beam scores；
- selected token IDs；
- parent beam indices；
- constraint node state；
- sequence history；
- Beam suffix KV ancestry。

如果只把 Trie 过滤放在 Worker，而把 select_token 留在 EngineCore，会形成：

~~~text
Worker 产生 W × K candidates
→ EngineCore 选择 token 和 parent
→ Worker 再执行 constraint state 与 KV reorder
~~~

这既扩大 IPC，也产生两套状态权威。因此采用一个 Worker 内部原子闭环：

~~~mermaid
flowchart LR
    LOGITS["Logits"]
    FILTER["Constraint Candidate Select"]
    LOCAL["Per-parent Top-K"]
    SCORE["Add Cumulative Beam Score"]
    GLOBAL["Global Top-W"]
    PARENT["Token IDs + Parent IDs"]
    STATE["Sequence + Constraint State"]
    KV["Beam KV Reorder"]
    DONE["Epoch Commit"]

    LOGITS --> FILTER
    FILTER --> LOCAL
    LOCAL --> SCORE
    SCORE --> GLOBAL
    GLOBAL --> PARENT
    PARENT --> STATE
    STATE --> KV
    KV --> DONE
~~~

这里的 select_token 应拆成三个清晰操作：

| 操作 | 含义 | 所有者 |
| --- | --- | --- |
| candidate_select | 在约束集合中为每个 parent 选择候选 | ConstraintHeadExtension |
| beam_select | 在 W × K 中为每个 item 选择新的 top-W | BeamSelector |
| final_select | 最后一步选择 result_width 个结果 | BeamSelector |

三个操作都位于 Worker 边界内。

### 6.2 Worker Session 状态

~~~python
@dataclass
class WorkerBatchBeamSession:
    batch_session_id: str
    child_request_ids: tuple[str, ...]  # B
    item_slots: tuple[int, ...]         # B
    resource_handle: str

    step: int
    epoch: int
    beam_scores: Tensor                 # [B, W]
    sequence: Tensor                    # [B, W, S]
    constraint_state: Tensor            # [B, W]
    active_mask: Tensor                 # [B, W]
    terminal_mask: Tensor               # [B]
~~~

Worker 是这些字段的唯一权威。EngineCore 不保存可独立重建另一套 Beam decision 的
scores、Trie node 或 sequence。

### 6.3 Worker 返回协议

Worker 只返回调度和最终路由需要的紧凑结果：

~~~python
@dataclass(frozen=True)
class WorkerBeamStepResult:
    batch_session_id: str
    step: int
    epoch: int
    selected_token_ids: Tensor   # [B, W]
    parent_indices: Tensor       # [B, W]
    active_mask: Tensor          # [B, W]
    terminal_mask: Tensor        # [B]
    finished: bool
    error: str | None
~~~

当结果返回 EngineCore 时，Worker 已完成 constraint state、sequence 和 Beam KV 的实际提交。
parent_indices 可以用于观测和 Scheduler 投影，但 EngineCore 不再重新执行选择。

---

## 7. 默认 Constraint Resource：模型级辅助产物

### 7.1 冻结决策

一个模型实例启动时只加载一个默认约束资源：

- 配置只声明一个 artifact；
- Worker 启动时加载一次；
- 所有 BatchBeamSession 共享同一个 immutable ResourceHandle；
- 请求可以省略 resource_id，由 Engine 注入默认 ID；
- 如果请求显式携带非默认 ID，admission 直接失败；
- 当前不支持运行期注册第二张表；
- 当前不支持逐出、热更新和按用户覆盖。

保留 resource_id 的目的，是稳定内部 ABI，而不是在当前版本开放多资源能力。

### 7.2 配置模型

推荐在 GR 的 VllmConfig 扩展中增加强类型 ConstraintHeadConfig。YAML 是输入形式，
Worker 不直接解析任意 YAML。

~~~yaml
constraint_head:
  enabled: true
  resource_id: default

  artifact:
    uri: /models/openonerec/constraint/trie.npz
    format: trie_csr_v1
    sha256: "<artifact-digest>"
    tokenizer_digest: "<tokenizer-digest>"
    vocab_size: "<model-vocab-size>"

  runtime:
    backend: cpu_trie
    placement: host
    required: true
~~~

内部配置：

~~~python
@dataclass(frozen=True)
class ConstraintArtifactSpec:
    uri: str
    format: str
    sha256: str
    tokenizer_digest: str
    vocab_size: int


@dataclass(frozen=True)
class ConstraintRuntimeConfig:
    backend: str
    placement: str
    required: bool


@dataclass(frozen=True)
class ConstraintHeadConfig:
    enabled: bool
    resource_id: str
    artifact: ConstraintArtifactSpec
    runtime: ConstraintRuntimeConfig
~~~

必须区分：

| 字段 | 表达什么 | 示例 |
| --- | --- | --- |
| format | 磁盘序列化格式 | trie_csr_v1、prefix_table_v1 |
| backend | 运行时执行器 | cpu_trie、cuda_table、npu_table |
| placement | 常驻位置 | host、device、hybrid |
| resource_id | 模型实例内部稳定标识 | default |

同一种 artifact format 可以被不同 backend 物化，避免文件格式与设备实现绑定。

### 7.3 加载时机

资源生命周期像权重，但不进入模型 state_dict：

~~~mermaid
flowchart TB
    YAML["YAML or Deployment Config"]
    TYPED["Typed ConstraintHeadConfig"]
    INIT["Worker.init_device"]
    MODEL["Load Model Weights"]
    HOST["Load and Validate Host Resource"]
    DEVICE["Optional Device Materialization"]
    BEAM["Initialize Beam Runtime Capacity"]
    PROFILE["Determine Available Memory"]
    KV["Allocate Native Paged KV"]
    WARM["Warmup and Graph Capture"]

    YAML --> TYPED
    TYPED --> INIT
    INIT --> MODEL
    MODEL --> HOST
    HOST --> DEVICE
    DEVICE --> BEAM
    BEAM --> PROFILE
    PROFILE --> KV
    KV --> WARM
~~~

推荐调用关系：

~~~python
def Worker.load_model(self):
    self.model_runner.load_model()
    self.model_runner.load_constraint_head(
        self.vllm_config.constraint_head_config
    )
    self.model_runner.initialize_beam_runtime(
        self.vllm_config.beam_runtime_config
    )
~~~

逻辑上三者都属于 Worker 启动阶段，但资源类型仍然分开：

- 模型参数属于 Weight Loader；
- Constraint Artifact 属于 Constraint Resource Loader；
- Beam KV 属于 Beam Runtime Pool。

如果 Device Constraint Buffer 或 Beam Pool 在显存探测后才创建，原生 Paged KV 可能已经
占用全部可用显存。因此所有常驻设备资源必须在 determine_available_memory 之前物化或预留。

### 7.4 Resource Manager 与 Buffer

~~~python
class ConstraintResourceManager:
    def load(
        self,
        config: ConstraintHeadConfig,
        model_meta: ModelMeta,
        device: Device,
    ) -> ConstraintResourceHandle:
        ...

    def get_default(self) -> ConstraintResourceHandle:
        ...

    def close(self) -> None:
        ...
~~~

~~~python
@dataclass(frozen=True)
class ConstraintResourceHandle:
    resource_id: str
    artifact_format: str
    backend: str
    digest: str
    host_resource: object | None
    device_resource: object | None
~~~

当前 CPU Trie：

~~~text
Host Resource
├─ node_offsets
├─ edge_token_ids
├─ edge_child_nodes
└─ terminal_nodes

Device Resource = None
~~~

未来 Device Table：

~~~text
Host Resource
└─ Manifest or mmap source

Device Resource
├─ token IDs
├─ offsets
├─ values
└─ backend metadata
~~~

Buffer 是 immutable persistent resource，不是可逐出的 request cache。它只在 Worker
shutdown、模型 unload 或显式重新加载模型时释放。

设备 Tensor 可以由 ResourceManager 直接持有，也可以在 ConstraintHeadExtension 中使用
non-persistent module buffer；无论哪种方式，都不能被写入模型 state_dict。

### 7.5 启动校验

以下任一条件失败都应终止 Worker 启动：

- artifact 不存在或不可读；
- format version 不支持；
- artifact digest 不匹配；
- tokenizer digest 不匹配；
- vocab_size 与 LM Head logits 宽度不匹配；
- CSR 或 table shape 不合法；
- TP/PP 相关 rank 得到不同资源版本；
- required resource 无法完成对应 placement 的物化。

不允许在配置声明 required 时静默回退到无约束生成。

---

## 8. Scheduler 与 BatchBeamCoordinator

### 8.1 EngineCore 保存协调状态，不保存 Beam 算法状态

~~~python
@dataclass
class BatchBeamCoordinator:
    batch_session_id: str
    child_request_ids: tuple[str, ...]
    stage: str
    prefill_ready_mask: tuple[bool, ...]
    next_epoch: int
    in_flight_epoch: int | None
    completed_epoch: int | None
    worker_session_handle: str | None
    final_result_sent: bool
~~~

它负责：

- 创建和回收 B 个 child Requests；
- 汇总 Prefill readiness；
- 将一个逻辑 step 拆成可执行 micro-batches；
- 检查 step 和 epoch；
- 处理 cancel、error 和 late output；
- 向 Frontend 投递一次终态。

它不负责：

- 读取 Trie；
- 计算 candidate scores；
- global top-W；
- 更新 Beam sequence；
- 执行 Beam KV reorder。

### 8.2 Decode continuation

Beam 的 W 个 lane 是同一逻辑时间步的并行行，不能追加为 child Request 的串行 token
history。Scheduler 需要显式 Beam continuation metadata：

~~~python
@dataclass(frozen=True)
class BeamContinuation:
    batch_session_id: str
    child_request_id: str
    item_index: int
    epoch: int
    step: int
    lane_token_ids: tuple[int, ...]
    parent_indices: tuple[int, ...]
    active_width: int
~~~

child Request 的 Scheduler 视图继续用于 Prefix KV 和 admission；真正的 Beam lane history
位于 Worker Session。

### 8.3 每一步仍经过 Scheduler

~~~text
Worker Step Result
→ BatchBeamCoordinator validates epoch
→ write next BeamContinuation
→ Scheduler.schedule
→ SchedulerOutput
→ Worker executes next step
~~~

Scheduler 继续负责：

- token budget；
- Prefix KV binding；
- ready/waiting 队列；
- physical batch composition；
- cancel 与 Request free；
- 执行是否能够下发。

### 8.4 调度计量

需要区分：

~~~text
logical_step_count = 1
execution_token_count = sum of scheduled active lanes
active_item_count = number of scheduled items
~~~

对固定 W 的场景，一个完整 Decode step 的理论执行行数为 B × W；如果 active mask
收缩或分 micro-batch，下发计量必须使用本轮真实 scheduled lanes。

---

## 9. Prefix Paged KV 与 Beam Suffix KV

### 9.1 两类 KV 的职责

~~~mermaid
flowchart LR
    PROMPT["Prompt Tokens"]
    NATIVE["Native Paged Prefix KV"]
    SHARE["Shared by W Beams"]
    DECODE["Beam Decode Tokens"]
    SUFFIX["Dedicated Beam Suffix KV"]
    SELECT["Parent-based KV Reorder"]

    PROMPT --> NATIVE
    NATIVE --> SHARE
    SHARE --> DECODE
    DECODE --> SUFFIX
    SUFFIX --> SELECT
    SELECT --> DECODE
~~~

| KV 类型 | 所有者 | 组织方式 | 生命周期 |
| --- | --- | --- | --- |
| Prefix KV | Scheduler / native KV manager | Paged KV | child Prefill 到 Session teardown |
| Beam suffix KV | Worker BeamKVManager | 固定容量或专用 Pool | Worker Session create 到 release |

Prefill Prefix 只保存一份，不为 W 个 Beam 复制。Beam 分叉只发生在 suffix。

### 9.2 容量与显存预算

所有持久和峰值内存必须满足：

~~~text
M_weights
+ M_constraint
+ M_nativeKV
+ M_beamKV
+ M_beamState
+ M_graph
+ M_workspace
+ M_temporary
+ M_fragmentation
<= M_usable
~~~

Beam runtime 配置至少描述：

~~~yaml
beam_runtime:
  max_active_beam_items: "<deployment-capacity>"
  max_beam_width: "<deployment-capacity>"
  max_decode_steps: "<deployment-capacity>"
  max_active_sessions: "<deployment-policy>"
~~~

这些值是资源 envelope，也是 Scheduler admission 的依据。不能在首个请求到达后按照请求
大小临时创建或扩容 Beam KV Pool。

CPU Trie 只占 Host 内存；未来 Device Table 会占设备常驻内存。一个模型实例只加载一个
默认资源，使这一部分显存可在启动期准确预算。

---

## 10. Session 状态、Epoch 与资源释放

### 10.1 Batch Session 状态机

~~~mermaid
stateDiagram-v2
    [*] --> WAITING
    WAITING --> PREFILLING: admit B child Requests
    PREFILLING --> PREFILL_READY: all items ready
    PREFILL_READY --> DECODING: Worker Session created
    DECODING --> DECODING: epoch committed
    DECODING --> FINALIZING: Worker reports terminal
    FINALIZING --> DRAINING: stop scheduling and teardown
    DRAINING --> FINISHED: Worker ack and child free
    FINISHED --> [*]

    WAITING --> ABORTED
    PREFILLING --> ABORTED
    PREFILL_READY --> ABORTED
    DECODING --> ABORTED
    WAITING --> FAILED
    PREFILLING --> FAILED
    PREFILL_READY --> FAILED
    DECODING --> FAILED
    ABORTED --> DRAINING
    FAILED --> DRAINING
~~~

### 10.2 Epoch 协议

每个 Session 初始最多一个 in-flight epoch：

~~~text
dispatch:
  require in_flight_epoch is None
  in_flight_epoch = next_epoch
  next_epoch += 1

complete:
  require result.epoch == in_flight_epoch
  completed_epoch = in_flight_epoch
  in_flight_epoch = None
~~~

如果一个 epoch 被拆成多个 micro-batches，in_flight_epoch 在全部分片被 Worker
StepAccumulator 接收并原子提交之前不能清空。

### 10.3 Cancel、错误和 teardown

资源释放顺序：

~~~mermaid
flowchart TD
    TERM["Terminal, Cancel or Failure"]
    STOP["Stop New Scheduling"]
    LATE{"Epoch In Flight?"}
    DRAIN["Drain or Fence Late Result"]
    FINAL["Publish One Final Result"]
    WFREE["Worker Releases Session and Beam KV"]
    ACK["Worker ReleaseAck"]
    CFREE["Scheduler Frees B Child Requests"]
    PFREE["Release Prefix KV Bindings"]
    REMOVE["Remove Coordinator Tombstone"]

    TERM --> STOP
    STOP --> LATE
    LATE -->|Yes| DRAIN
    LATE -->|No| WFREE
    DRAIN --> WFREE
    STOP --> FINAL
    WFREE --> ACK
    ACK --> CFREE
    CFREE --> PFREE
    FINAL --> REMOVE
    PFREE --> REMOVE
~~~

最终结果可以先于物理资源释放返回，但 Coordinator tombstone 必须保留到：

- final result 已投递；
- Worker session 已释放；
- B 个 child Requests 已 free；
- Prefix KV binding 已解除。

约束资源不随 Session teardown 释放，它继续作为模型实例的常驻资源。

---

## 11. Online Serving 与 Offline API

Online 与 Offline 是两个平级 Frontend：

~~~mermaid
flowchart TB
    subgraph FRONTENDS["Peer Frontends"]
        direction LR
        ONLINE["Online Serving"]
        OFFLINE["Offline API"]
    end

    API["EngineBeamAPI"]
    CORE["BatchBeamCoordinator + Scheduler"]
    WORKER["Worker BeamRuntime"]

    ONLINE --> API
    OFFLINE --> API
    API --> CORE
    CORE --> WORKER
~~~

两者共用：

- BatchBeamSession；
- B child Request 模型；
- 原生 Prefill 与 Prefix KV；
- Worker Constraint Resource；
- BeamRuntime 与 BeamKV；
- Scheduler continuation；
- final-only 内部协议。

入口差异只体现在：

| 维度 | Online Serving | Offline API |
| --- | --- | --- |
| 到达方式 | 动态请求 | 业务可提前组好 B |
| 优化目标 | 延迟与取消响应 | 吞吐与 batch 完成时间 |
| admission policy | 按到达和容量 | 可利用已知 batch 信息 |
| 输出适配 | OpenAI 或 RequestOutput | list、tensor 或 batch result |

Offline 不建立另一套 Beam 主循环，也不采用“先把所有全局 Prefill 做完，再统一 Decode”的
独立执行器。它仍然使用同一个 Scheduler，只是提交时已经知道完整业务 Batch。

---

## 12. TP、PP 与 DP 下的资源和选择

约束资源在逻辑上只有一个默认版本，但在分布式模型中可能存在多个物理副本。

### 12.1 Selector Rank

推荐由拥有完整 logits 的 selector rank 执行：

- Constraint candidate select；
- global Beam select；
- final select。

随后广播：

~~~text
selected_token_ids
parent_indices
active_mask
terminal_mask
~~~

所有持有 Beam suffix KV 的 TP/PP ranks 根据相同 parent_indices 执行 KV reorder。

### 12.2 Resource Placement

- 所有相关 Worker 校验相同 artifact digest；
- CPU Trie 可以只在 selector rank 常驻；
- Device Table 只在执行约束算子的 rank 物化，除非算子本身是分布式实现；
- 每个 DP replica 拥有本地物理副本；
- BatchBeamSession 必须对 DP replica 保持粘性，不能在 Decode 中途迁移。

---

## 13. CPU Trie 到 Device Table 的演进

上层只依赖统一 ConstraintExecutor：

~~~python
class ConstraintExecutor:
    def select_candidates(
        self,
        logits: Tensor,
        constraint_state: Tensor,
        resource: ConstraintResourceHandle,
        top_k: int,
    ) -> ConstraintSelection:
        ...
~~~

### 13.1 当前 backend

~~~text
artifact format = trie_csr_v1
runtime backend = cpu_trie
placement = host
~~~

CPU Trie 作为 Worker 内部实现继续使用。即使发生 Device 到 Host 的候选同步，也不再把
候选发送到 EngineCore。

### 13.2 后续 backend

~~~text
artifact format = trie_csr_v1 or prefix_table_v1
runtime backend = cuda_table or npu_table
placement = device
~~~

替换后：

- Config、ResourceHandle 和 Session ABI 保持稳定；
- Scheduler 不感知 Trie 或 Device Table；
- EngineCore 不重新获得 candidate decision；
- BeamSelector 和 BeamKVManager 继续使用相同 selected token 与 parent 协议；
- 常驻 Buffer 地址可以纳入后续 graph capture。

### 13.3 多资源能力的重访条件

当前不实现多个 resource_id。只有出现以下明确需求时再扩展：

- 同一模型实例必须同时服务多个独立 catalog；
- 重启模型切换 catalog 的成本不可接受；
- 多资源带来的额外 Host/Device 内存有明确预算；
- Session 能够绑定资源 generation，并完成并发版本释放。

扩展时可以把单个 default handle 演进为只读 registry，而不改变 Request 中预留的
resource_id 字段。

---

## 14. 最终设计决策

1. Online Serving 与 Offline API 是平级入口，共用一套 Engine 和 Worker Runtime。
2. 一个业务 BatchRequest 对应一个 BatchBeamSession。
3. 一个 BatchBeamSession 由 B 个原生 child Requests 支撑，不合成一个原生 Request。
4. B 个 child Requests 独立使用原生 Scheduler、Paged KV、Prefix Cache 和 chunked
   prefill。
5. Batch Prefill barrier 完成后进入共同 Beam Decode。
6. Prefix KV 继续由 Scheduler 持有，Beam suffix KV 由 Worker BeamKVManager 持有。
7. EngineCore 的算法 Controller 收敛为 BatchBeamCoordinator，只保留生命周期与调度状态。
8. Worker 是 Beam score、sequence、constraint state 和 Beam KV 的唯一权威。
9. candidate_select、beam_select、select_token、final_select 和 KV reorder 全部位于 Worker。
10. 一个逻辑 step 可以拆成多个物理 micro-batches，但只能原子提交一次 epoch。
11. 每个 Decode execution 继续经过 Scheduler，EngineCore 不绕过 Scheduler 调用 Worker。
12. 一个模型实例只加载一个默认约束资源，内部保留 resource_id。
13. 约束资源由 ConstraintHeadConfig 声明，在 Worker 模型加载阶段解析并常驻。
14. 约束资源不进入 state_dict，不随 Request 发送，不参与 Session 逐出。
15. 当前采用 CPU Trie backend，未来通过相同接口替换为 Device Table。
16. Device Constraint Buffer 和 Beam Pool 在原生 KV 显存探测前完成物化或预留。
17. Frontend 只接收一次 BatchBeamFinalResult，不消费中间候选。
18. 当前不涉及用户关联 KV、约束资源热更新和多资源 registry。

---

## 15. 相关设计

### 15.1 背景

- [#35：Serving Beam Controller 下沉与 Persistent Beam Session](https://github.com/zhanghanleo10/vllm-gr/issues/35)
- [PR #290：Engine-owned Controller PoC](https://github.com/JiusiServe/vllm-gr/pull/290)

### 15.2 本仓库相关文档

- [Beam 增量 Decode 统一架构](./beam_incremental_decode_unified_architecture_design.md)
- [BeamKV Cache 架构与调度](./beam_kv_cache_architecture_and_scheduling_design.md)
- [近线 GR 多 Batch 调度策略](./nearline_batch_scheduler_and_input_protocol_design.md)
- [Serving 与 EngineCore 传输边界](./vllm_serving_enginecore_shared_memory_analysis.md)
- [Beam Decode KV Buffer 组织分析](./beam_decode_kv_buffer_organization_analysis.md)
