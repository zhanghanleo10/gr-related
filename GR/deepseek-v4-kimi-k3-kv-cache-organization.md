# DeepSeek V4 与 Kimi K3：vLLM KV Cache 组织图

> 更新日期：2026-08-17  
> 阅读顺序：每张图从 ① 到 ④；先看数据是什么，再看 block/page，最后看显存池。

---

## 1. DeepSeek V4

![DeepSeek V4 KV Cache：从单层语义、1000 token 分页、Group Block Table 到 packed backing](assets/deepseek-v4-kv-cache-layout.svg)

### 图中最重要的三句话

1. **C4 的“64”是一个 page 中的压缩 entry 数，不是只覆盖 64 个原始 token。**一个 C4 Main page 对应 256 个原始 token：`256 / 4 = 64 entries`。
2. **Compressor state 逻辑上单独成 Group、领取独立 Block ID；物理上仍来自同一个 packed backing。**它不会塞进 Main KV 的 block。
3. **全局 Block ID 没有固定 token 数语义。**C4 Main、C128 Main、SWA、Indexer 和 state 各自通过 cache spec/block table 解释同一个编号空间。

### 1000 token 对照表

| Cache | 原始 token 语义 | 实际保存 |
| --- | --- | --- |
| C4 Main KV | `0..999` | `floor(1000/4) = 250` 个 compressed latent entries |
| C4 Indexer KV | `0..999` | 250 个 128-D retrieval keys |
| C128 Main KV | `0..999` | `floor(1000/128) = 7` 个 compressed latent entries |
| Main / Indexer state | 当前未完成压缩的尾部 | rolling state，不保存 1000 份历史 |
| SWA KV | 最近 128 个有效 token：`872..999` | 64-token page 对齐后占 `[832..895]`、`[896..959]`、`[960..1023]` |

SWA 从 832 开始，是因为 872 所在的 64-token page 必须向下对齐：

```text
floor(872 / 64) × 64 = 832
```

其中 `832..871` 被 mask，`1000..1023` 尚未写入。

### 不同大小的 page 如何共用池

```text
group_row_bytes[g] = Σ page_bytes(layer in group g)
block_stride       = max_g(group_row_bytes[g])
```

- 每个 Group 有独立 block table 和 page 语义。
- `KVCacheCoordinator` 只建立一个全局 `BlockPool`。
- worker 申请一个 `[num_global_blocks, block_stride]` packed backing。
- 一个 Block ID 同时只归一个 Group；Group 内各 layer page 使用该行不同 offset。
- page-size grouping / layer regrouping 用于减少行内 padding；共享全局池避免独立池容量被困住。

---

## 2. Kimi K3

![Kimi K3 KV Cache：69 个固定状态 KDA 层与 24 个 Paged MLA 层](assets/kimi-k3-kv-cache-layout.svg)

### 图中最重要的三句话

1. **K3 不是 93 层都保存逐 token KV。**69 个 KDA 层只保存固定 conv/recurrent state。
2. **只有 24 个 Gated MLA 层随序列增长。**每 token 保存 `512 latent + 64 PE tail = 576` 个元素。
3. **KDA 与 MLA 是两个 cache group。**Prefix caching 时必须取两者都能恢复的共同最长边界。

### 单层、单 TP rank 的保存内容

| Layer type | Cache 内容 | 是否随 `L` 增长 |
| --- | --- | --- |
| KDA | conv state：`[3, 36864/TP]`，通常 BF16 | 否 |
| KDA | recurrent state：`[96/TP, 128, 128]`，FP32 | 否 |
| Gated MLA | 每 token 576 elements；默认 BF16 为 1152 B/token/layer | 是 |

### 1000 token

```text
K3_cache(1000)
  ≈ 69 × fixed_KDA_state_per_layer_per_rank
  + 24 × 1000 × 576 × bytes_per_element
  + page padding / metadata
```

- 69 个 KDA 层：长度从 1 增至 1000，仍是 69 份固定 rolling state，而不是 69 × 1000 份 KV。
- 24 个 MLA 层：每层 1000 个 latent entries，即 `ceil(1000 / B)` 个 page。
- BF16 下 24 层有效 MLA payload 约为 `24 × 1000 × 1152 B = 27.65 MB`，未计 page padding/metadata。

---

## 3. GR 为什么选择固定 BeamKV Buffer，而不是继续使用 Paged KV

### 3.1 决策范围

这里不是要替换 vLLM 的全部 Paged KV，而是按照数据语义拆成两部分：

| 数据 | 特征 | 选择 |
| --- | --- | --- |
| Prompt KV | 1K–10K、长度可变、所有 Beam 共享 | 保留 Native Paged KV，通过 block refcount 共享 |
| Beam suffix KV | Beam width 64–512、只有 2–5 步、每轮按 Parent 重排 | 使用预分配的 Dense BeamKV Arena |
| `parent_ids`、`active_mask`、`decode_step` | 每轮变化，但 shape 有上界 | 使用固定地址的 Device Buffer |
| Compressor/KDA 一类可变状态 | 一个状态代表整段历史 | 独立 slot；Beam 重排时必须随 Parent select/copy |

推荐结构：

```text
Shared Prompt Paged KV
        │
        ├── Beam 0 ─┐
        ├── Beam 1 ─┼── Dense BeamKV[capacity, max_steps, ...]
        └── Beam W ─┘             │
                              parent_ids select
```

核心判断是：**长、变长、跨请求共享的数据适合 Paged；短、定长、频繁 Parent 重排的数据更适合 Dense Buffer。**

### 3.2 当前场景为什么不同于通用在线 Decode

选择 Dense BeamKV 依赖以下约束，而不是假设 Dense 永远优于 Paged：

| 约束 | GR 场景 |
| --- | --- |
| Beam width | 固定为 64–512，不支持动态 Beam width |
| Decode depth | 固定且很短，通常 2–5 步 |
| Prompt | 长且由所有 Beam 共享 |
| Suffix | Session 内私有，不需要跨请求 Prefix Cache |
| 每轮状态变化 | Parent→Child 重排 + 写入一个新 token |
| 执行目标 | 增量 forward，并逐步支持 Full CUDA/ACL Graph |

Paged KV 的核心价值是处理未知长度、动态增长、请求 churn 和跨请求 prefix sharing。上述能力对 Prompt 很重要，但对容量已知的 2–5 token Beam suffix 价值有限。

### 3.3 Per-Beam Paged：短尾部的 page 利用率很低

如果每个 Beam 都拥有自己的 Paged suffix，设：

```text
Beam width W = 256
Decode depth D = 5
Paged block size B = 16
```

Dense Buffer 需要的有效 token slots：

```text
W × D = 256 × 5 = 1280
```

每个 Beam 至少占用一个 16-token page：

```text
W × ceil(D / B) × B
= 256 × 1 × 16
= 4096 token slots
```

此时 suffix page 利用率只有：

```text
1280 / 4096 = 31.25%
```

| Beam width | Dense：`W × 5` | Per-Beam Paged：`W × 16` | Paged / Dense |
| ---: | ---: | ---: | ---: |
| 64 | 320 slots | 1024 slots | 3.2× |
| 256 | 1280 slots | 4096 slots | 3.2× |
| 512 | 2560 slots | 8192 slots | 3.2× |

此外还会增加每个 Beam 的 block-table entry、slot mapping、allocator/refcount bookkeeping，以及 Parent 分叉时的 block ownership/COW 处理。

### 3.4 Grouped Paged：碎片较小，但 Beam 身份不稳定

另一种办法是把所有 Beam suffix 拼成一个 grouped request：

```text
Round 1: b0:s0    | b1:s0    | ... | bW:s0
Round 2: b0:s0,s1 | b1:s0,s1 | ... | bW:s0,s1
Round 3: b0:s0..s2| b1:s0..s2| ... | bW:s0..s2
```

这种组织可以把多个短 suffix 紧密装入 page，因此不能简单地说“每个 Beam 都浪费一个 page”。它真正的问题是：

```text
grouped_position(beam, step)
    = beam × current_suffix_len + step
```

`current_suffix_len` 每轮增长，导致：

- Beam 的 grouped logical position 跨轮变化；
- Paged KV 仍以 token sequence 为中心，不能直接表达稳定的 Beam identity；
- Parent→Child 关系需要重建 grouped row、物理 select/copy，或额外的 ancestry/COW mapping；
- 当前路径若每轮重提完整 suffix，就会重复 forward 历史 token；
- 如果 Attention 需要连续 suffix，Paged suffix 还可能需要额外 gather。

因此两种 Paged 方案的主要问题不同：

| 方案 | 优点 | 主要问题 |
| --- | --- | --- |
| Per-Beam Paged | Beam/request 语义直接 | 2–5 token 的 page tail 浪费严重，metadata 多 |
| Grouped Paged | page 利用率高 | Beam identity/Parent 语义和跨轮位置不稳定 |

### 3.5 当前 Grouped 路径的重复 Forward

如果第 `r` 轮重新提交每条 Beam 的完整 suffix，则 `D` 轮总 forward token 数为：

```text
Grouped replay = W × D × (D + 1) / 2
Incremental     = W × D
Replay ratio    = (D + 1) / 2
```

| Decode depth `D` | Grouped replay / Incremental |
| ---: | ---: |
| 2 | 1.5× |
| 3 | 2× |
| 5 | 3× |

例如 `W=256, D=5`：

```text
Grouped replay = 256 × 15 = 3840 token forwards
Incremental     = 256 × 5  = 1280 token forwards
```

需要准确说明：**重复 forward 不是 Paged KV 的理论必然结果，而是当前 grouped request 缺少持久 Beam identity 和历史复用机制的结果。**可以在 Paged KV 上继续增加 ancestry index/COW，但这相当于再实现一套 Beam 专用间接寻址机制。

### 3.6 Dense BeamKV 为什么更匹配 Parent 语义

Dense Arena 可以直接组织成：

```text
BeamKV[layer, beam_capacity, max_steps, ...]
```

稳定地址为：

```text
slot(beam, step) = beam × max_steps + step
```

Parent 重排可以明确表示为：

```text
new_beam[b, 0:step]
    = old_beam[parent_ids[b], 0:step]
```

由此得到：

- 每轮只 forward `W` 个新增 token；
- Beam slot 跨轮保持稳定；
- Parent select 可以实现为固定 shape 的 device kernel；
- suffix attention 可直接读取连续布局；
- 不需要为 2–5 token suffix 维护 page allocation/hash/refcount；
- Session 结束时整体归还 Beam slot。

### 3.7 Full Graph 是收益，但不是唯一理由

Paged KV 本身也可以进入 Full Graph，只要：

```text
paged backing 地址固定
block_table 地址固定
slot_mapping 地址固定
每轮只更新这些静态 Device Tensor 的内容
```

因此不能把决策表述成“Paged KV 不能成图”。更准确的说法是：

> Dense BeamKV 减少了 replay 热路径中的 block allocation、page-boundary handling、grouped-position rebuild 和动态 metadata，使固定 capacity 的 Beam decode 更容易满足 Full Graph contract。

固定 BeamKV 带来的 Graph 收益包括：

- `data_ptr`、shape、stride 和 workspace 可在 capture 前固定；
- `parent_ids`、`active_mask`、`decode_step` 使用固定地址 Device Buffer；
- 不在 replay 内申请 suffix block；
- 每个 Beam capacity gear 只需捕获一次；
- Parent 不同只改变 Buffer 内容，不要求重新 capture。

但仅有固定 Buffer 仍不保证 Full Graph：Python data-dependent control flow、运行时 Tensor/workspace 分配、不支持 capture 的算子、D2H 同步和错误的跨 Stream 依赖仍然会破图。

### 3.8 方案对比与最终选择

| 维度 | Paged Prompt KV | Per-Beam Paged suffix | Grouped Paged suffix | Dense BeamKV suffix |
| --- | --- | --- | --- | --- |
| 长度弹性 | 强 | 强，但当前不需要 | 强，但当前不需要 | 按固定 `max_steps` 预留 |
| Prompt/跨请求共享 | 强 | suffix 价值低 | suffix 价值低 | 不提供 |
| 2–5 token 空间利用率 | 不适用 | 低 | 高 | 高 |
| 稳定 Beam identity | 不适用 | 有 | 弱 | 强 |
| Parent 重排 | 不适用 | block/COW 管理 | 需重建或 ancestry mapping | 直接 device select/copy |
| 增量 forward | 支持 | 可支持 | 当前路径不支持 | 天然支持 |
| Full Graph | 可以 | 可以但 metadata 多 | 可以但布局更新复杂 | contract 最简单 |
| 适用结论 | 保留 | 不推荐 | 可演进但复杂 | 当前 GR 推荐 |

最终采用：

```text
Native Paged KV for long shared Prompt
                  +
Fixed-capacity Dense BeamKV for short mutable suffix
                  +
Device-side parent selection between rounds
```

这是针对当前 workload 的选择，不是对 Paged KV 的通用否定。

### 3.9 Dense 方案的代价与重评条件

Dense BeamKV 同样有明确代价：

- 按 `beam_capacity × max_steps` 提前保留容量；
- capacity padding 可能浪费空间；
- Parent select 会产生有界的 copy/gather 流量；
- 需要维护 Paged Prompt + Dense suffix 两套 Attention 输入；
- 不适合长 Decode、动态 Beam width 或跨请求 suffix sharing。

出现以下情况时应重新评估 Paged/ancestry-index 方案：

- Decode depth 明显增长；
- 引入动态 Beam width；
- Dense capacity 预留成为主要显存开销；
- Parent select 成为可观的显存带宽瓶颈；
- 出现成熟的 ancestry-index Attention，能够避免物理 KV 重排；
- Beam suffix 开始具有实际的跨请求共享价值。

核心边界：**共享 Prompt 不等于共享可变 Beam state；Paged 可以成图，但对固定 W、极短 suffix 和高频 Parent 重排并不是最小充分的数据结构。**

---

## 4. 参考实现

- [vLLM：DeepSeek-V4 在线推理系统设计](https://vllm.ai/blog/2026-04-24-deepseek-v4)
- [DeepSeek V4 attention API](https://docs.vllm.ai/en/v0.27.1/api/vllm/models/deepseek_v4/attention/)
- [`kv_cache_utils.py`：异构 spec 分组与 packed layout](https://github.com/vllm-project/vllm/blob/fe1c317157d4478fdc0e02096447e61305b871e9/vllm/v1/core/kv_cache_utils.py)
- [`kv_cache_coordinator.py`：全局 BlockPool 与共同 prefix 命中](https://github.com/vllm-project/vllm/blob/fe1c317157d4478fdc0e02096447e61305b871e9/vllm/v1/core/kv_cache_coordinator.py)
- [`attn_utils.py`：worker packed backing 与 layer view](https://github.com/vllm-project/vllm/blob/fe1c317157d4478fdc0e02096447e61305b871e9/vllm/v1/worker/gpu/attn_utils.py)
- [vLLM：Kimi K3 Preview](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-07-22-kimi-k3-preview.md)
- [vLLM：Kimi K3 正式支持](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-07-27-k3.md)
- [Kimi K3 tracking issue](https://github.com/vllm-project/vllm/issues/50001)
