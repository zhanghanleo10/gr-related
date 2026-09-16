"""Real-weight, multiprocess OneRec capture; each profiler gets W10 + N5."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODEL = Path('/home/z00469465/OneRec/OneRec-1.7B')
CATALOG = Path('/home/z00469465/OneRec/generated_data/video_constraint_triples.json')
PROMPT = ROOT / 'tests/resources/single_one_rec_prompt.txt'

if os.environ.get('ONEREC_V1_TRACE_ADAPTER') == '1':
    from v1_trace_adapter import install
    install()  # Also executed when multiprocessing spawn imports this script.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiler', choices=['nsys', 'torch'], required=True)
    parser.add_argument('--mode', choices=['v1', 'legacy'], default='v1')
    args = parser.parse_args()
    out = Path(__file__).resolve().parent / args.profiler
    out.mkdir(exist_ok=True)
    os.environ['VLLM_CUSTOM_SCOPES_FOR_PROFILING'] = str(int(args.profiler == 'torch'))
    os.environ['VLLM_NVTX_SCOPES_FOR_PROFILING'] = str(int(args.profiler == 'nsys'))
    import torch
    import vllm
    import vllm_gr
    import nvtx
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    assert Path(vllm_gr.__file__).resolve().parent.parent == ROOT
    profiler_config = {'profiler': 'cuda'} if args.profiler == 'nsys' else {
        'profiler': 'torch', 'torch_profiler_dir': str(out),
        'torch_profiler_with_stack': True, 'torch_profiler_record_shapes': True,
        'torch_profiler_with_memory': True, 'torch_profiler_use_gzip': False,
        'torch_profiler_dump_cuda_time_total': True, 'ignore_frontend': True,
    }
    config = dict(
        model=str(MODEL), dtype='bfloat16', seed=0,
        async_scheduling=True, trust_remote_code=True,
        max_logprobs=128, catalog_path=str(CATALOG), constraint_backend='constraint_table',
        beam_execution_mode=args.mode, beam_graph_enabled=True,
        beam_max_width=128, beam_max_decode_steps=3,
        attention_config={'backend': 'CUSTOM'}, max_num_seqs=1, max_model_len=8192,
        enable_prefix_caching=True, gpu_memory_utilization=0.7,
        profiler_config=profiler_config,
    )
    if args.mode == 'v1':
        for key in ('catalog_path', 'constraint_backend', 'beam_execution_mode',
                    'beam_graph_enabled', 'beam_max_width', 'beam_max_decode_steps',
                    'attention_config'):
            config.pop(key)
        config['vllm_gr_config'] = json.loads((Path(__file__).resolve().parent / 'gr_config.json').read_text())
    params = BeamSearchParams(
        beam_width=128, max_tokens=5, temperature=0.0, ignore_eos=False,
        begin_token='<|sid_begin|>', end_token='<|sid_end|>',
    )
    catalog = {(str(r['a']), str(r['b']), str(r['c'])) for r in json.loads(CATALOG.read_text())['triples']}
    init_start = time.perf_counter()
    with GRLLM(**config) as llm:
        init_s = time.perf_counter() - init_start
        tokenizer = llm.get_tokenizer()
        source = tokenizer.encode(PROMPT.read_text(), add_special_tokens=False)
        newline = tokenizer.encode('\n', add_special_tokens=False)
        assert len(newline) == 1
        ids = source[-1023:] if len(source) >= 1023 else newline * (1023 - len(source)) + source
        assert len(ids) == 1023
        prompts = [{'prompt_token_ids': ids}]
        begin = tokenizer.convert_tokens_to_ids('<|sid_begin|>')
        end = tokenizer.convert_tokens_to_ids('<|sid_end|>')
        os.environ['ONEREC_TRACE_BEGIN_ID'] = str(begin)
        os.environ['ONEREC_TRACE_END_ID'] = str(end)
        method = llm.beam_search_v1 if args.mode == 'v1' else llm.beam_search

        def validate(output):
            assert len(output) == 1 and len(output[0].sequences) == 128
            rows = []
            for seq in output[0].sequences:
                tokens = list(seq.tokens)
                score = float(seq.cum_logprob)
                assert len(tokens) == 1028 and tokens[:1023] == ids
                assert tokens[1023] == begin and tokens[-1] == end
                sid = tokens[1024:-1]
                assert len(sid) == 3 and math.isfinite(score)
                assert tuple(tokenizer.convert_ids_to_tokens(sid)) in catalog
                rows.append({'sid_token_ids': sid, 'score': score})
            assert all(a['score'] >= b['score'] - 1e-7 for a,b in zip(rows, rows[1:]))
            return {'beam_count': 128, 'catalog_legal': True,
                    'sha256': hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest(),
                    'rows': rows}

        def stats():
            return llm.llm_engine.collective_rpc('get_gr_beam_graph_runtime_stats')

        def reset():
            assert llm.reset_prefix_cache(), 'prefix-cache reset failed'

        warmups = []
        before_warmup = stats()
        for index in range(10):
            reset()
            start = time.perf_counter_ns()
            result = method(prompts, params)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            validation = validate(result)
            warmups.append({'call_ms': elapsed, 'sha256': validation['sha256']})
            print(f'WARMUP {index + 1}/10 {elapsed:.3f} ms', flush=True)
        after_warmup = stats()
        raw_outputs, samples = [], []
        frontend = None
        if args.profiler == 'torch':
            frontend = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                with_stack=True, record_shapes=True, with_modules=True,
            )
            frontend.start()
        llm.start_profile('onerec_b1_bw128_l1024_sid3_w10_n5')
        try:
            for index in range(5):
                label = f'onerec_request_{index + 1:02d}'
                reset()
                scope = nvtx.annotate(label, domain='onerec') if args.profiler == 'nsys' else torch.profiler.record_function(label)
                with scope:
                    start = time.perf_counter_ns()
                    raw_outputs.append(method(prompts, params))
                    samples.append({'request': index + 1, 'call_ms': (time.perf_counter_ns() - start) / 1e6})
        finally:
            llm.stop_profile()
            if frontend is not None:
                frontend.stop()
                frontend.export_chrome_trace(str(out / 'frontend.pt.trace.json'))
        after_profile = stats()
        validations = [validate(output) for output in raw_outputs]
        result = {
            'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'engine_config': config, 'init_seconds': init_s,
            'effective_engine_config': str(llm.llm_engine.vllm_config),
            'versions': {'python': sys.version, 'torch': torch.__version__, 'vllm': vllm.__version__},
            'source': {'vllm': vllm.__file__, 'vllm_gr': vllm_gr.__file__},
            'contract': {'batch_size': 1, 'beam_width': 128, 'sid_steps': 3,
                         'submitted_prompt_tokens': 1023, 'model_input_tokens': 1024,
                         'warmups': 10, 'profiled_requests': 5,
                         'multiprocessing': True, 'async_scheduling': True,
                         'prefix_cache_reset_before_every_request': True,
                         'explicit_per_request_cuda_synchronize': False,
                         'output_validation_outside_capture': True},
            'test_only_final_result_adapter': os.environ.get('ONEREC_V1_TRACE_ADAPTER') == '1',
            'final_result_adapter_scope': 'TEST_ONLY_FINAL_RESULT_ADAPTER',
            'environment': {k:v for k,v in os.environ.items() if k.startswith(('VLLM_', 'CUDA_', 'ONEREC_', 'OMP_', 'TORCHINDUCTOR_'))},
            'weights': [{'name': p.name, 'bytes': p.stat().st_size} for p in MODEL.glob('*.safetensors')],
            'prompt_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            'warmups': warmups, 'samples': samples, 'validation': validations,
            'all_outputs_identical': len({v['sha256'] for v in validations} | {w['sha256'] for w in warmups}) == 1,
            'graph_stats': {'before_warmup': before_warmup, 'after_warmup': after_warmup, 'after_profile': after_profile},
        }
        (out / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'samples': samples, 'all_outputs_identical': result['all_outputs_identical'], 'graph_stats': result['graph_stats']}), flush=True)
    print('CAPTURE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
