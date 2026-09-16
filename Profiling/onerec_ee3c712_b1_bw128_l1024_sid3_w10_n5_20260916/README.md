# OneRec-1.7B 离线 beam_search_v1 全流程 trace

当前源码提交：`ee3c7129a47f9db688880f163dd810206e8e9b9d`。
调用入口是 **`GRLLM.beam_search_v1()`**。只采集 V1；无 HTTP 服务。
本目录内的脚本、配置和产物是本次新增内容，`vllm_gr` 和已安装 vLLM 源码均未修改。

## 固定场景

- OneRec-1.7B 真实 safetensors 权重，BF16，物理 GPU0 / NVIDIA L20。
- batch=1，BW=128，生成 SID=3；模型输入 1024 token，即 1023 prompt + `<|sid_begin|>`。
- canonical CUDA constraint table，CUSTOM attention，temperature=0，length_penalty=1。
- 多进程 EngineCore，`async_scheduling=True`，保留原生调度与 CUDA Graph。每次请求前 reset prefix cache，实际执行完整 1K prefill。
- Nsight、PyTorch 各自独立启动引擎，各自预热 **10 次**，然后连续采集 **5 次**。初始化、图捕获、warmup 不在正式窗口。
- 正式请求范围：`onerec_request_01` … `onerec_request_05`。不在每个请求末尾额外插入 CUDA synchronize。输出一致性与 catalog 校验在采集窗口外。

## 末尾适配器的边界

当前提交的 `BeamSearchOutputProcessor.ensure_available()` 为未实现占位，Worker 正常完成时也尚未打包 `BeamBatchResult`。按用户选定的方案，`v1_trace_adapter.py` 只在采集进程中补齐这个最终结果消费接口：

1. 原生 `AsyncGPUBeamOutput.get_output()` 等待最终 control snapshot 后，读取真实 GPU completed/history 状态，按已有测试的回溯规则得到 SID 和累计分数，选取前 128 个结果，返回协议对象。
2. 前端封装测试结果对象，添加输入及 SID 起止标记供场景校验。起止标记属于显示封装，3 个 SID 均来自实际模型及 Beam GPU 状态。

没有替换 Scheduler、Executor、forward、Beam 候选选择、KV 管理、中间步骤或释放逻辑。没有使用 legacy `beam_search()`，也没有用假结果绕过模型。

适配器在两份 trace 中明确标记为：

- `TEST_ONLY_FINAL_RESULT_ADAPTER/worker_D2H_sort_pack`
- `TEST_ONLY_FINAL_RESULT_ADAPTER/frontend_wrap`

**该末尾区间是测试适配，不代表尚未实现的原生 A5 收尾性能。** 概览图用浅橙色背景标记此区间。全栈 tracing 也会扰动 CPU 发射时序；本次带 profiler 的时长不作为无 profiler 的延迟基准。

## 查看产物

- `full_cpu_gpu.pt.trace.json`：合并前端和 Worker 的全量 PyTorch Chrome trace，可直接用 Perfetto / Chrome tracing 打开；`.gz` 为压缩版。
- `nsys/full_trace.nsys-rep`：用 Nsight Systems 打开，包含 CPU 原生采样栈、Python 栈、原生 vLLM NVTX 范围、CUDA API 及调用栈、CUDA Graph 内部 kernel、GPU 拷贝与 OS runtime 事件。
- `cpu_gpu_overlap.png` / `.svg`：5 次请求的 CPU/GPU 重叠概览。CPU 范围包含等待，不等于 CPU 持续计算；GPU busy union 与 kernel 耗时相加值不是同一指标。
- `nsys/full_trace.sqlite`：Nsight 完整结构化导出。
- `torch/`：未经合并的前端/Worker 原始 trace、算子表、运行配置、输出校验和日志。
- `nsys_trace_validation.json` / `torch_trace_validation.json`：事件类别、原生调用覆盖及逐请求检查。
- `environment.json`：权重、tokenizer、输入、catalog 的 SHA-256 与依赖版本。

Nsight 每个请求实际记录 1,876 个 GPU kernel、129 次拷贝、3 次 CUDA Graph launch，5 次共 9,380 个 kernel 和 645 次拷贝。开启 `--cuda-trace-all-apis` 后同一 launch 有带版本和不带版本两条 API 记录，统计已按线程和 correlation ID 去重。

`graph_stats.full_dispatches` 是旧计数器，在 V1 本次路径保持 0；实际 graph replay 以 trace 中去重的 15 次 launch 为准。

## 复现

从仓库根目录运行：

```bash
bash trace/onerec_ee3c712_b1_bw128_l1024_sid3_w10_n5_20260916/run_capture.sh
```

脚本依次运行 Nsight 和 PyTorch，输出目录需改为新目录，避免覆盖已有产物。只采 PyTorch 可传入 `torch` 参数。分析与合并：

```bash
python trace/onerec_ee3c712_b1_bw128_l1024_sid3_w10_n5_20260916/analyze_nsys.py
python trace/onerec_ee3c712_b1_bw128_l1024_sid3_w10_n5_20260916/merge_torch.py
```

合并依据两个 Kineto 文件的 `baseTimeNanoseconds` 对齐时间轴，保留所有事件与调用栈；flow ID 按来源独立重编号并保留数值格式，stack ID 添加来源前缀，避免冲突。不混合 Nsight 和 PyTorch 两次独立运行的时间轴。

本次合并 trace 共 247,950 个事件，包含 217,507 个 Python 调用事件和 9,800 个 GPU kernel 事件；其中 87,463 个 Python 调用事件来自原生 vLLM。Nsight 和 PyTorch 两轮正式请求的结果摘要均为 `e2dbdafdab23c59b1384e9df456c2bf701d36b9d4671aabe98934cfea93862d2`。
