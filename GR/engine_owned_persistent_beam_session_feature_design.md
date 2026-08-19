# vLLM-GR Engine-Owned Persistent Beam Session 大特性方案

> 状态：Feature Design  
> 日期：2026-08-19  
> 目标仓库：JiusiServe/vllm-gr  
> 主 RFC：[Serving Beam Controller 下沉与 Persistent Beam Session](https://github.com/zhanghanleo10/vllm-gr/issues/35)  
> 当前 PoC：[Move the Serving beam controller into EngineCore](https://github.com/JiusiServe/vllm-gr/pull/290)  
> 范围：Online Serving 与 Offline API 共用的 Engine-owned Beam Controller、Persistent Request、Scheduler Continuation、约束资源、终态协议和生命周期闭环  
> 非目标：Worker/GPU Beam decision、静态 suffix KV Pool、CUDA/ACL Full Graph、Width Bucket、普通请求混部

---

## 0. 一页结论

当前 vllm-gr Beam Search 的根本问题，不是某个 Beam helper 慢，也不是一次 IPC 消息太大，而是
**Beam Session 的控制权放错了位置**：

~~~text
Serving 计算下一步 Beam
→ 发起一次 Engine 请求
→ Scheduler / Worker 执行一步
→ 中间结果返回 Serving
→ Serving 再计算下一步
~~~

一个本应在 Engine 内连续存在的 Session，被拆成了多次 Frontend ↔ Engine 往返。由此产生：

- Serving 持有 Beam 算法状态和逐步循环；
- 每一步都可能重建 Request；
- Prefix KV 所有权与释放时机不稳定；
- Scheduler 无法成为 step、epoch 和资源生命周期的权威；
- Online 与 Offline 容易复制两套 Beam 主循环；
- 后续接入 Worker Beam、独立 BeamKV 和 Full Graph 时还要再次重写控制流。

本方案的核心决策是：

> **Serving 和 Offline 只负责入口适配；Beam Session 由 EngineCore 持有；同一个 Session 只
> admission 一次；Prefill 和所有 Decode step 复用同一个逻辑 Request；每一步仍然经过
> Scheduler；中间结果留在 Engine 内，Frontend 只接收一次终态。**

目标主链如下：

~~~mermaid
flowchart TB
    subgraph ENTRY["Peer Frontends"]
        direction LR
        ON["Online Serving Adapter"]
        OFF["Offline API Adapter"]
    end
    API["Engine Beam API"]
    CTRL["BeamSearchController"]
    SCH["Scheduler + Persistent Request"]
    WRK["Worker / ModelRunner"]
    FINAL["BeamFinalResult"]

    ON --> API
    OFF --> API
    API --> CTRL
    API --> SCH
    SCH --> WRK
    WRK --> CTRL
    CTRL --> SCH
    CTRL --> FINAL
    FINAL --> ON
    FINAL --> OFF
~~~

总体方案由五层组成：

| 层次 | 核心职责 |
| --- | --- |
| Frontend Adapter | Online/Offline 参数适配与最终格式化 |
| Engine Control | Session registry、Controller 与终态路由 |
| Scheduler Lifecycle | Persistent Request、continuation 与 Prefix KV 所有权 |
| Shared Resource | Immutable constraint resource 等跨请求资源 |
| Execution | 继续复用 Scheduler → Worker → ModelRunner 主链 |

---

## 1. 这项特性真正要解决什么

### 1.1 当前链路

当前 Serving 不只是协议适配层，它还承担：

- 逐 token Beam 主循环；
- active/completed beams；
- parent/child/token routing；
- candidate extraction 和 constraint filtering；
- EOS、Top-K 和 parent selection；
- history 与 logprobs 回溯；
- 每一步 Engine 请求的创建；
- 最终排序与输出构造。

~~~mermaid
sequenceDiagram
    participant C as Client
    participant S as Serving
    participant E as EngineCore
    participant H as Scheduler
    participant W as Worker

    C->>S: Beam request
    S->>E: Prefill request
    E->>H: schedule
    H->>W: Prefill
    W-->>S: raw output
    loop Every Beam step
        S->>S: TopK / constraint / parent decision
        S->>E: step update + new request
        E->>H: schedule
        H->>W: one decode step
        W-->>S: intermediate output
    end
    S->>S: sort / history / format
    S-->>C: final result
~~~

这条链路看起来可以工作，但它破坏了三个基本边界：

1. **算法边界**：Beam decision 属于推理状态机，不属于 HTTP/OpenAI 协议层。
2. **资源边界**：Request 和 KV 的生命周期必须由 Scheduler/Engine 负责。
3. **执行边界**：一个 Session 的中间 step 不应该反复穿过 Frontend 边界。

### 1.2 目标不是“少发几条消息”

仅把多条消息压缩成一条消息，不能解决所有权问题。即使 Serving 和 Engine 在同一进程，只要
Serving 仍持有 Beam 主循环，以下问题仍然存在：

- Offline 入口必须复制算法；
- Cancel 与 Engine 资源释放无法原子闭环；
- Scheduler 看不到完整 Session 进度；
- Worker Beam 状态与 Controller 生命周期无法绑定；
- Graph、BeamKV Pool 和 session slot 无法建立稳定 lease。

因此本特性的成功标准不是“IPC 次数下降”，而是：

> **即使删掉 OpenAI Serving，Engine 仍能独立完成一个 Beam Session。**

---

## 2. 约束与设计不变量

### 2.1 硬约束

以下约束决定了最小架构：

1. **稳定 Session ID**：一个外部 Beam 请求对应一个稳定的 beam session。
2. **单次 admission**：同一个 Session 只创建并准入一个逻辑 Request。
3. **Scheduler 权威**：任何 Prefill/Decode execution 都由 Scheduler 产生。
4. **单 step in-flight**：同一个 Session 同时最多一个 epoch 在执行。
5. **Prefix KV 常驻**：Prefill 完成不进入普通 finished/free 路径。
6. **中间结果内收**：Frontend 不消费 Prefill/Decode 中间输出。
7. **终态唯一**：FINISHED、ABORTED、FAILED 只发布一次最终结果。
8. **算法等价**：第一阶段只移动所有权，不改变 Beam 结果。
9. **Frontend 无关**：Online 与 Offline 调用同一个 Engine Beam API。
10. **不依赖 Graph**：控制流正确性必须先在 eager 路径成立。

### 2.2 需要挑战的假设

| 常见假设 | 判断 | 原因 |
| --- | --- | --- |
| “Controller 下沉后可以直接调用 Worker” | 不成立 | 会绕过 Scheduler 的资源和进度权威 |
| “每步创建新 Request 更简单” | 局部简单、系统错误 | Prefix KV、身份、取消和释放都变得不稳定 |
| “active width 个 token 可以追加到 Request history” | 不成立 | 它们是并行 lane，不是串行时间步 |
| “Online 和 Offline 应完全走同一种 policy” | 不成立 | 二者共享机制，但优化目标不同 |
| “Constraint catalog path 可以直接发给 Engine” | 不成立 | Engine 不应依赖 Frontend 文件系统和 tokenizer 对象 |
| “run-to-completion 等于最终 Offline 策略” | 不成立 | 它是第一阶段 policy，不是机制限制 |

---

## 3. 特性范围与演进边界

### 3.1 本特性负责

~~~mermaid
flowchart LR
    F["Frontend adapters"] --> C["Engine control plane"]
    C --> S["Scheduler lifecycle"]
    S --> X["Existing eager execution"]
    X --> R["One final result"]
~~~

具体包括：

- 抽取纯 Beam Controller；
- Controller 在 EngineCore 的注册、路由和清理；
- per-step Request 过渡路径；
- Persistent Request 与 Prefix KV 保持；
- step/epoch/in-flight 协议；
- immutable Trie 资源注册；
- final-only 结果协议；
- Online/Offline 共用 Engine API；
- Cancel、错误和迟到结果闭环；
- 可观测性与 parity tests。

### 3.2 本特性不负责

下列能力在此方案之上演进，但不能成为本特性的合入前置：

- Top-K、Beam group select 和 history 全量下沉到 Worker/GPU；
- 独立 suffix BeamKV Pool；
- Worker 持久 Beam execution slot；
- fixed shape、Width Bucket 和 active mask；
- CUDA/ACL Full Graph；
- TP/PP Beam decision 广播；
- 多 Session 矩形 Full Graph；
- 普通生成请求与 GR 请求混部；
- PD 分离和 Prefix KV 跨节点传输。

这条边界非常重要：

> 本特性建立“谁拥有 Session、Request 如何持续、何时完成和释放”；BeamKV 与 Graph 方案建立
> “设备内数据放在哪里、怎样执行得更快”。前者是后者的控制面基础，但两者不应合并成一个 PR。

---

## 4. 总体架构

### 4.1 逻辑组件

~~~mermaid
flowchart TB
    subgraph FRONTEND["Frontend"]
        ONLINE["Online Serving Adapter"]
        OFFLINE["Offline API Adapter"]
    end

    subgraph ENGINE["EngineCore"]
        API["EngineBeamAPI"]
        REG["BeamSessionRegistry"]
        CTRL["BeamSearchController"]
        CON["ConstraintResourceRegistry"]
        OUT["FinalResultRouter"]
    end

    subgraph SCHED["Scheduler"]
        REQ["Persistent Beam Request"]
        META["BeamContinuation"]
        KV["Native Prefix KV Lease"]
    end

    subgraph EXEC["Worker / ModelRunner"]
        MODEL["Model Forward"]
        RAW["Step Raw Output"]
    end

    ONLINE --> API
    OFFLINE --> API
    API --> REG
    REG --> CTRL
    CON --> CTRL
    API --> REQ
    REQ --> META
    META --> MODEL
    MODEL --> RAW
    RAW --> CTRL
    CTRL --> META
    CTRL --> OUT
    OUT --> ONLINE
    OUT --> OFFLINE
    REQ --> KV
~~~

### 4.2 责任矩阵

| 组件 | 持有什么 | 不应该持有什么 |
| --- | --- | --- |
| Online Adapter | 参数校验、tokenize、Cancel、OpenAI 格式化 | Beam 主循环、active beams、KV |
| Offline Adapter | 批量提交、离线结果适配、job-level abort | 独立 Controller、逐步 Worker 调用 |
| EngineBeamAPI | 规范化请求、Session 创建、最终结果 future | HTTP、tokenizer、文件路径 |
| BeamSessionRegistry | Controller、context、terminal tombstone | Paged KV block 分配 |
| BeamSearchController | Beam 算法状态、scores、history、constraint state | Scheduler queue、KV allocator |
| ConstraintResourceRegistry | digest → immutable Trie | tokenizer、catalog path |
| Scheduler | Request、stage、epoch、budget、Prefix KV | 全量 Beam history 与最终协议 |
| Worker | 当前 step 的执行输入与 raw output | Frontend callback、Session 终态权威 |
| FinalResultRouter | 一次性投递、frontend correlation | 下一步 Beam decision |

### 4.3 统一 Engine API

Online 和 Offline 必须收敛到同一个内部入口：

~~~python
@dataclass(frozen=True)
class EngineBeamRequest:
    session_id: str
    external_request_id: str
    prompt_token_ids: tuple[int, ...]
    beam_width: int
    max_tokens: int
    eos_token_id: int
    sampling: BeamSamplingSpec
    constraint_resource_id: str | None
    output_spec: BeamOutputSpec
    frontend_kind: str

class EngineBeamAPI:
    def add_beam_request(request: EngineBeamRequest) -> BeamResultFuture: ...
    def cancel_beam_request(session_id: str) -> bool: ...
~~~

关键点：

- API 接受 token IDs，不接受 tokenizer；
- API 接受 resource ID，不接受 catalog path；
- API 返回一个最终结果 future，不暴露每步 raw output；
- Online 与 Offline 的差异保留在 adapter 和 scheduling policy；
- Controller、Request、KV 和 final contract 完全共用。

---

## 5. Online Serving 与 Offline API：同等 Frontend

Online Serving 和 Offline API 是两个平级入口。二者都直接调用 EngineBeamAPI，任何一方都不
通过另一方间接进入 Engine，也不在自己的入口层保存 Beam 主循环。

~~~mermaid
flowchart TB
    subgraph ENTRY["Peer Frontends"]
        ON["Online Serving"]
        OFF["Offline API"]
    end

    API["EngineBeamAPI"]
    CORE["Engine-owned Beam Session"]
    SCH["Scheduler + Persistent Request"]
    EXEC["Worker / ModelRunner"]

    ON --> API
    OFF --> API
    API --> CORE
    CORE --> SCH
    SCH --> EXEC
~~~

### 5.1 共用内核

| 能力 | Online Serving | Offline API |
| --- | --- | --- |
| EngineBeamAPI | 共用 | 共用 |
| BeamSearchController | 共用 | 共用 |
| Persistent Request | 共用 | 共用 |
| Prefix KV lifecycle | 共用 | 共用 |
| Scheduler continuation | 共用 | 共用 |
| Constraint registry | 共用 | 共用 |
| BeamFinalResult | 共用 | 共用 |

### 5.2 入口差异

| 维度 | Online Serving | Offline API |
| --- | --- | --- |
| 调用方式 | 动态请求 | 单请求或请求列表 |
| 优化目标 | P99、取消响应、请求隔离 | 吞吐、批量完成时间 |
| admission | 按在线到达准入 | 可利用已知 batch 信息 |
| scheduling policy | 可采用 run-to-completion | 可采用 multi-session cohort |
| 输出适配 | OpenAI RequestOutput | list、tensor 或 batch result |
| 失败语义 | 单请求终态 | 单 Session 隔离与 job policy |

这些差异只属于 Adapter 和 SchedulingPolicy，不改变 Session、Controller、Request 或 Worker
执行协议。

### 5.3 Offline 多 Session

~~~mermaid
flowchart TB
    B["Offline request list"]
    A["Batch admission"]
    R1["Session A persistent request"]
    R2["Session B persistent request"]
    R3["Session C persistent request"]
    G["Scheduler ready set"]
    X["Shape-compatible execution batch"]
    O["Independent final results"]

    B --> A
    A --> R1
    A --> R2
    A --> R3
    R1 --> G
    R2 --> G
    R3 --> G
    G --> X
    X --> R1
    X --> R2
    X --> R3
    R1 --> O
    R2 --> O
    R3 --> O
~~~

每个 Session 保持独立的 Controller、Request、Prefix KV、step、epoch 和终态。多个 Session
可以组成一个物理 execution batch，但不能被合并成一个逻辑 Request。

### 5.4 Policy 接口

~~~python
class BeamSchedulingPolicy(Protocol):
    def select_ready_sessions(
        self,
        ready: Sequence[PersistentBeamRequestState],
        budget: BeamExecutionBudget,
    ) -> Sequence[str]: ...
~~~

run_to_completion 与 offline_cohort 只改变 Scheduler 如何选择 ready Session，不改变
EngineBeamAPI、Persistent Request 和 Worker ABI。

---

## 6. 五个核心抽象

### 6.1 BeamSearchController：纯算法状态机

Controller 的职责是把一个 step raw output 变成下一步 decision：

~~~text
Controller state + StepRawOutput
              ↓
      BeamControllerDecision
              ↓
 next tokens / parents / active width / finished
~~~

推荐接口：

~~~python
class BeamSearchController:
    def consume_output(
        self,
        output: BeamControllerStepOutput,
    ) -> BeamControllerDecision: ...

    def cancel(self) -> None: ...

    def finalize(self) -> BeamFinalResult: ...
~~~

它持有：

- active beams 与 completed beams；
- cumulative scores；
- parent/history 信息；
- current step；
- constraint node states；
- EOS/stop/token budget；
- Beam 算法自身 metrics。

它不持有：

- Scheduler Request；
- Paged KV block；
- Worker slot；
- OpenAI RequestOutput；
- Engine output queue。

这是整个方案最重要的抽象边界。只有 Controller 足够纯，才能：

- 用旧 Serving 实现做 golden parity；
- 在 EngineCore、单测或未来 Worker provider 中复用；
- 独立替换 CPU/Numpy 算法；
- 不改变 Scheduler 主循环。

### 6.2 BeamControllerDecision：算法与调度之间的窄协议

~~~python
@dataclass(frozen=True)
class BeamControllerDecision:
    completed_step: int
    next_tokens: tuple[int, ...]
    parent_beam_indices: tuple[int, ...]
    active_width: int
    finished: bool
    finish_reason: str | None
    error: BeamError | None
~~~

该对象表达“下一步做什么”，但不表达“如何分配 KV”或“调用哪个 Worker”。

必须满足：

- next_tokens 长度等于 active_width；
- parent index 位于上一轮有效 lane 范围；
- completed_step 与 Scheduler expected step 一致；
- finished 后 next_tokens 必须为空；
- 同一个 completed_step 不能被消费两次。

### 6.3 PersistentBeamRequestState：调度生命周期

Controller state 与 Scheduler state 是两套正交状态，不应塞进同一个巨型对象。

~~~python
@dataclass
class PersistentBeamRequestState:
    session_id: str
    stage: BeamRequestStage
    logical_step: int
    next_epoch: int
    in_flight_epoch: int | None
    completed_epoch: int | None
    execution_width: int
    active_width: int
    prefix_binding: PrefixKVBinding
    final_result_sent: bool
    scheduler_freed: bool
    worker_teardown_sent: bool
~~~

它只关心：

- Request 是否可调度；
- 当前是否已有 step 在执行；
- 下一次 SchedulerOutput 应带什么 metadata；
- Prefix KV 是否仍被 Session 持有；
- 终态资源是否已经释放。

### 6.4 Immutable Constraint Resource：跨边界资源

Catalog/Trie 不应该跟随每个请求反复发送，也不应该让 Engine 打开 Frontend 的文件路径。

推荐流程：

~~~mermaid
flowchart LR
    CAT["Catalog + Tokenizer"]
    COMP["Frontend Compiler"]
    TRIE["Immutable Token-ID Trie"]
    DIG["Digest"]
    REG["Engine Registry"]
    REQ["Beam Request"]

    CAT --> COMP
    COMP --> TRIE
    TRIE --> DIG
    TRIE --> REG
    DIG --> REQ
    REQ --> REG
~~~

资源必须具备：

- token-ID based；
- immutable；
- contiguous/validated arrays；
- versioned wire format；
- content digest；
- idempotent registration；
- unknown digest fail-fast。

这不仅解决跨进程问题，也为未来把 Trie 上传到设备侧建立稳定资源 ID。

### 6.5 BeamFinalResult：内部终态协议

内部结果与 OpenAI/Offline 输出必须分层：

~~~python
@dataclass(frozen=True)
class BeamFinalResult:
    session_id: str
    external_request_id: str
    beams: tuple[BeamFinalBeam, ...]
    finish_reason: str
    error: BeamError | None
    metrics: BeamSessionMetrics | None
~~~

配套的 BeamOutputSpec 决定：

- result width；
- logprobs = none / selected / top-k；
- top logprobs 数量；
- 是否需要 full history。

这样可以避免默认把所有候选、所有 step、所有 logprobs 发送回 Frontend。

---

## 7. 双状态机：算法状态与资源状态分离

### 7.1 Controller 状态

~~~mermaid
stateDiagram-v2
    [*] --> RUNNING
    RUNNING --> RUNNING: consume step
    RUNNING --> COMPLETED: search finished
    RUNNING --> CANCELLED: cancel
    RUNNING --> ERROR: invalid output
    COMPLETED --> FINALIZED: build final result
    CANCELLED --> FINALIZED: build cancel result
    ERROR --> FINALIZED: build error result
    FINALIZED --> [*]
~~~

### 7.2 Persistent Request 状态

~~~mermaid
stateDiagram-v2
    [*] --> PREFILLING
    PREFILLING --> PARKED: prefill output consumed
    PARKED --> BEAM_DECODING: continuation scheduled
    BEAM_DECODING --> PARKED: epoch completed
    PARKED --> FINALIZING: controller finished
    BEAM_DECODING --> FINALIZING: terminal output
    FINALIZING --> DRAINING: teardown dispatched
    DRAINING --> FINISHED: scheduler and worker safe
    FINISHED --> [*]
    PREFILLING --> FAILED
    PARKED --> FAILED
    BEAM_DECODING --> FAILED
    FINALIZING --> FAILED
    FAILED --> DRAINING
~~~

PARKED 的语义不是 Request 结束，而是：

> 当前没有 execution in flight，Prefix KV 和 Request 仍然存活，等待 Controller 提供下一步
> continuation。

### 7.3 为什么需要两个状态机

Controller 可能已经 COMPLETED，但 Worker teardown 仍未完成；Scheduler 也可能已经进入
DRAINING，但 FinalResult 已经构造完成。因此：

- Controller terminal 不等于资源已释放；
- Final result ready 不等于可以删除 tombstone；
- Scheduler free 不等于 Worker slot 已安全回收；
- Cancel 不等于可以忽略 in-flight completion。

如果用一个 boolean finished 表达全部状态，就无法正确处理迟到结果与幂等清理。

---

## 8. Online 与 Offline 统一控制流

~~~mermaid
sequenceDiagram
    participant ON as Online Serving
    participant OFF as Offline API
    participant E as EngineCore
    participant B as Beam Controller
    participant H as Scheduler
    participant W as Worker

    alt Online request
        ON->>E: add_beam_request once
    else Offline request
        OFF->>E: add_beam_request once
    end
    E->>B: create controller
    E->>H: admit persistent request
    H->>W: Prefill
    W-->>E: raw prefill output
    E->>B: consume output
    B-->>E: next decision
    loop Until terminal
        E->>H: continue same request
        H->>W: one Beam decode step
        W-->>E: raw step output
        E->>B: consume output
        B-->>E: decision
    end
    E->>H: finalize and drain
    alt Online result
        E-->>ON: one BeamFinalResult
    else Offline result
        E-->>OFF: one BeamFinalResult
    end
~~~

无论入口是 Online 还是 Offline，Frontend 边界都只发生两次业务事件：

1. 单次 submit；
2. 单次 final。

Cancel 是第三种控制事件，但不是逐步数据路径。

---

## 9. Persistent Request 的最大实现难点

### 9.1 一个逻辑 Request，不等于一条串行 token 序列

普通生成 Request 的 token history 是：

~~~text
t0 → t1 → t2 → t3
~~~

Beam step 的输入是同一时间步的并行 lane：

~~~text
step k:
lane 0 token
lane 1 token
lane 2 token
...
lane W-1 token
~~~

因此不能把 active_width 个 lane token 追加成：

~~~text
... → lane0 → lane1 → lane2
~~~

否则 Scheduler 会误以为 Request 串行前进了 active_width 个 token，继而污染：

- computed token count；
- position；
- block hash；
- prefix-cache identity；
- token budget；
- KV block allocation；
- 下一步 request history。

### 9.2 分离三种“长度”

~~~mermaid
flowchart TB
    L["Logical progress"]
    E["Execution lanes"]
    K["KV materialization"]

    L --> A["Beam generation step +1"]
    E --> B["This forward executes W rows"]
    K --> C["Prefix KV retained; suffix follows Beam layout"]
~~~

需要明确区分：

| 概念 | 含义 | 示例 |
| --- | --- | --- |
| logical_step | Beam 算法推进次数 | 第 2 个 Decode step |
| execution_width | 本轮物理执行 lane 数 | 128 rows |
| active_width | 本轮有效 Beam 数 | 93 beams |
| prefix_len | 共享 Prompt 有效长度 | 900 tokens |
| suffix_step | Beam suffix 已物化的时间步 | 2 steps |

推荐调度计量：

~~~text
logical_step_count = 1
execution_token_count = execution_width
active_token_count = active_width
~~~

### 9.3 BeamContinuation：不要把 lane metadata 塞进逻辑历史

推荐增加 Engine 内部 typed continuation：

~~~python
@dataclass(frozen=True)
class BeamContinuation:
    session_id: str
    epoch: int
    completed_step: int
    lane_token_ids: tuple[int, ...]
    parent_beam_indices: tuple[int, ...]
    execution_width: int
    active_width: int
~~~

第一阶段可以继续复用上游 StreamingUpdate 来唤醒 PARKED Request，但它只能是兼容适配器：

- 先恢复稳定 prefix view；
- lane tokens 只进入当前 execution view；
- 不把 lane tokens 留在串行 prompt history；
- continuation 后恢复 prefix block hashes；
- 每轮验证 Request identity 和 Prefix block IDs；
- 禁止同一个 epoch 重复 schedule。

长期应把 BeamContinuation 变成 SchedulerOutput 的显式扩展，而不是长期依赖通用 streaming
session 的隐含语义。

### 9.4 Epoch 协议

调度条件必须可以机械验证：

~~~text
can_dispatch
  = stage is PARKED
  and in_flight_epoch is None
  and completed_epoch == next_epoch - 1
~~~

下发时：

~~~text
in_flight_epoch = next_epoch
next_epoch += 1
~~~

完成时：

~~~text
assert output.epoch == in_flight_epoch
completed_epoch = in_flight_epoch
in_flight_epoch = None
~~~

任何 mismatch 都 fail-fast，不允许：

- 静默重试同一步；
- 丢弃 mismatch 后继续；
- 回退到 Serving loop；
- 生成第二个 in-flight step。

---

## 10. Prefix KV 所有权

### 10.1 生命周期

~~~mermaid
flowchart LR
    A["Request admitted"]
    P["Prefill allocates Prefix KV"]
    D["Many Beam decode steps"]
    T["Terminal state"]
    F["Free once"]

    A --> P
    P --> D
    D --> D
    D --> T
    T --> F
~~~

硬规则：

- PREFILLING → PARKED 不释放；
- PARKED → BEAM_DECODING 不重新申请；
- 每个 continuation 使用相同 logical Request；
- FINISHED、ABORTED、FAILED 才进入最终释放；
- Request free、Controller remove、Worker teardown 分别幂等；
- Prefix block IDs 或等价 binding 在 Session 内保持稳定。

### 10.2 Scheduler 必须保留权威

EngineCore 可以驱动 Controller，但不能直接构造 Worker input：

~~~text
Controller decision
→ write BeamContinuation
→ Scheduler.schedule
→ SchedulerOutput
→ Worker
~~~

原因不是代码风格，而是 Scheduler 独占以下事实：

- 请求是否仍然合法；
- token/KV/sequence budget 是否足够；
- 当前是否可下发；
- Request 是否正在被取消；
- 哪些资源属于本轮；
- execution 完成后怎样更新和释放。

绕过 Scheduler 会让 Persistent Request 只剩名字，实际资源仍然由两套控制流管理。

---

## 11. Constraint Resource 的正确抽象

### 11.1 为什么 Catalog 不能是 Request payload

直接在 Request 中携带 catalog path 或 tokenizer 会引入：

- Frontend/Engine 文件系统不一致；
- tokenizer Python 对象不可移植；
- 每个请求重复编译 Trie；
- 大对象重复序列化；
- 无法建立设备资源缓存；
- 内容更新和版本语义不明确。

### 11.2 Compile once，register by digest

~~~mermaid
sequenceDiagram
    participant F as Frontend
    participant R as Engine Resource Registry
    participant C as Beam Controller

    F->>F: tokenize and compile catalog
    F->>F: compute digest
    F->>R: ensure immutable Trie
    R-->>F: registered or already exists
    F->>C: request with resource digest
    C->>R: require digest
    R-->>C: immutable Trie view
~~~

必须定义：

- resource format version；
- dtype、shape 与 CSR invariants；
- edge token 排序；
- node/child bounds；
- digest 覆盖全部语义字段；
- registration 幂等；
- shutdown 清理；
- 未知或冲突 digest 直接失败。

未来 GPU Trie buffer 可继续复用同一个 digest 和 lease 语义，而不改变 Request API。

---

## 12. Cancel、错误、迟到结果与清理

### 12.1 Session 终止不是一个函数调用

一个 Session 可能同时存在于：

- Controller registry；
- Scheduler request table；
- KV manager；
- Worker Beam session/slot；
- Engine output correlation table；
- in-flight device execution；
- terminal tombstone。

因此终止需要一个有序、幂等的协议：

~~~mermaid
flowchart TD
    X["Cancel or failure"]
    C["Stop Controller decisions"]
    S["Mark Request terminal"]
    R["Publish one final result"]
    I{"Step in flight?"}
    D["Drain late completion"]
    W["Dispatch Worker teardown"]
    K["Free Scheduler / KV once"]
    Z["Remove tombstone after both complete"]

    X --> C --> S
    S --> R
    S --> I
    I -->|Yes| D
    I -->|No| W
    D --> W
    W --> K
    R --> Z
    K --> Z
~~~

FinalResult 的发布不必等待设备资源全部回收，否则会把 teardown latency 暴露给用户；但
Session tombstone 只能在“终态已投递”和“资源已安全回收”两个条件都满足后删除。

### 12.2 迟到结果

Cancel 后到达的 output 不能：

- 恢复 Controller；
- 产生下一 decision；
- 再次发布终态；
- 重新创建 Request；
- 释放已经复用给其他 Session 的 slot。

需要保留有界 tombstone：

~~~text
session_id → terminal reason + last epoch + resource generation
~~~

直到确定不会再收到旧 epoch，或 Worker teardown ack 已完成。

### 12.3 错误域

推荐错误分类：

| 错误 | 处理 |
| --- | --- |
| 参数或资源 ID 无效 | admission 前拒绝 |
| Controller invariant 失败 | 当前 Session FAILED |
| step/epoch mismatch | 当前 Session fail-fast |
| Worker session-local error | 当前 Session FAILED |
| Scheduler 全局不变量损坏 | Engine fatal |
| 未知 active session output | 记录并隔离，必要时 Engine fatal |
| 已终止 Session 的迟到 output | tombstone 校验后丢弃 |

Offline batch 中，一个 Session 的合法 session-local failure 不应破坏其他 Session。

---

## 13. 输出协议与 Logprobs

### 13.1 Final-only

EngineCore 应拦截 Controller-owned 中间输出，不把它们放入 Frontend 普通 output queue。

Frontend 只看到：

~~~text
BeamFinalResult
  ├─ beams
  ├─ scores
  ├─ finish reason
  ├─ optional logprobs
  └─ optional structured error
~~~

### 13.2 Logprobs 瘦身

输出成本应由 caller-visible contract 决定：

| 模式 | 返回内容 | 适用场景 |
| --- | --- | --- |
| none | token、score、finish reason | 默认 Serving/Offline |
| selected | 最终选中 token 的 logprob | 评估与调试 |
| top-k | 每步有限候选 | 深度诊断 |
| full internal | 仅测试/开发 | 不作为默认公网协议 |

不要在每一步复制完整 history 和候选 logprobs。推荐：

1. Controller 内保存 parent pointer；
2. 只在最终 top result_width beams 上回溯；
3. 按 output spec 投影 logprobs；
4. 最终一次序列化。

---

## 14. 与后续执行优化的关系

本方案解决的是控制权、Request 生命周期与前后端边界。后续可以在不改变这些上层契约的
前提下继续演进执行层：

~~~mermaid
flowchart TB
    C["Engine-owned Beam Session"]
    S["Persistent Scheduler Contract"]
    W["Worker-owned Beam State"]
    K["Dedicated BeamKV Pool"]
    G["CUDA / ACL Full Graph"]

    C --> S
    S --> W
    W --> K
    K --> G
~~~

- Controller 可以从 EngineCore CPU 实现替换为 Worker/GPU decision；
- Prefix KV 仍由 Scheduler Request 持有；
- suffix BeamKV 可以迁移到固定 Pool；
- BeamContinuation 可以映射到固定地址的 device metadata；
- Full Graph 只改变一步怎样执行，不改变 Session 如何开始、继续和结束。

因此，Worker Beam、BeamKV 与 Graph 都是本方案之上的执行优化，而不是重新设计
Online/Offline 入口或 Scheduler 生命周期。

---

## 15. 最终设计决策

1. Beam Search 是 Engine 内部的长生命周期 Session，而不是 Serving coroutine。
2. BeamSearchController 首先以纯 CPU 算法对象形式下沉，行为与旧路径严格对齐。
3. EngineCore 驱动 Controller，但所有 model execution 仍经过 Scheduler。
4. Prefill 与 Decode 复用同一个逻辑 Request，Prefix KV 持有到 Session 终态。
5. Beam 并行 lane 与串行 token history 必须在数据模型中分离。
6. 每个 Session 同时最多一个 in-flight epoch，step/epoch mismatch fail-fast。
7. Constraint 以 immutable token-ID resource 注册，Request 只携带 digest。
8. 中间结果留在 Engine，Frontend 只接收一次 BeamFinalResult。
9. Cancel、错误、迟到 output、Scheduler free 与 Worker teardown 使用幂等闭环。
10. Online 与 Offline 共用 Engine API、Controller、Request 和资源生命周期。
11. Online 与 Offline 可以使用不同 SchedulingPolicy。
12. 当前 run-to-completion 和单 active slot 是可替换 policy，不是长期架构边界。
13. Output 与 Logprobs 使用独立内部协议，不侵入 Controller 和 Scheduler 生命周期。
14. Worker/GPU Beam decision、BeamKV 和 Graph 在该控制面之上独立演进。

---

## 16. 相关设计

### 16.1 背景

- [#35：Serving Beam Controller 下沉与 Persistent Beam Session](https://github.com/zhanghanleo10/vllm-gr/issues/35)
- [PR #290：Engine-owned Controller PoC](https://github.com/JiusiServe/vllm-gr/pull/290)

### 16.2 本仓库相关文档

- [Beam 增量 Decode 统一架构](./beam_incremental_decode_unified_architecture_design.md)
- [BeamKV Cache 架构与调度](./beam_kv_cache_architecture_and_scheduling_design.md)
- [近线 GR 多 Batch 调度策略](./nearline_batch_scheduler_and_input_protocol_design.md)
- [Serving 与 EngineCore 传输边界](./vllm_serving_enginecore_shared_memory_analysis.md)
- [Beam Decode KV Buffer 组织分析](./beam_decode_kv_buffer_organization_analysis.md)
