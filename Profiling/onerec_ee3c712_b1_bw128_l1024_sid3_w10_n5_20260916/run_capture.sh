#!/usr/bin/env bash
set -euo pipefail
cd /home/z00469465/PROJECT/vllm-gr
capture_dir=trace/onerec_ee3c712_b1_bw128_l1024_sid3_w10_n5_20260916
export CUDA_VISIBLE_DEVICES=0 USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENABLE_V1_MULTIPROCESSING=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export VLLM_ENABLE_PREFILL_CUDAGRAPH=1
export ONEREC_V1_TRACE_ADAPTER=1
export PYTHONPATH=/home/z00469465/PROJECT/vllm-gr
mkdir -p "$capture_dir/nsys" "$capture_dir/torch"
if [ ! -f "$capture_dir/gr_config.json" ]; then
    python "$capture_dir/prepare_config.py" > "$capture_dir/prepare_config.log" 2>&1
fi
if [ "${1:-all}" != torch ]; then
nsys profile --trace=cuda,nvtx,osrt --sample=process-tree --backtrace=dwarf \
    --python-sampling=true --python-sampling-frequency=1000 \
    --cudabacktrace=all:0 --python-backtrace=cuda \
    --cuda-graph-trace=node --cuda-trace-all-apis=true \
    --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none \
    --wait=all --export=sqlite \
    --python-functions-trace="$capture_dir/nsys_annotations.json" \
    --output="$capture_dir/nsys/full_trace" \
    python "$capture_dir/capture.py" --profiler nsys --mode v1 \
    > "$capture_dir/nsys/capture.log" 2>&1
fi
python "$capture_dir/capture.py" --profiler torch --mode v1 \
    > "$capture_dir/torch/capture.log" 2>&1
