"""Validate capture coverage and plot the actual Nsight CPU/GPU time axis."""
import json
import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
db = HERE / 'nsys/full_trace.sqlite'
con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
con.row_factory = sqlite3.Row
tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
strings = dict(con.execute('SELECT id,value FROM StringIds'))
nvtx = [dict(r) for r in con.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start')]
for row in nvtx:
    row['name'] = row['text'] or strings.get(row['textId'], '')
requests = [r for r in nvtx if r['name'].startswith('onerec_request_')]
assert len(requests) == 5, [(r['name'], r['start']) for r in requests]
kernels = [dict(r) for r in con.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start')]
copies = [dict(r) for r in con.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start')]
apis = [dict(r) for r in con.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start')]
for row in apis:
    row['name'] = strings[row['nameId']]
copy_types = {r['id']:r['label'] for r in con.execute('SELECT * FROM ENUM_CUDA_MEMCPY_OPER')}


def clipped(rows, lo, hi):
    return [(max(r['start'], lo), min(r['end'], hi)) for r in rows if r['start'] < hi and r['end'] > lo]


def union_ns(intervals):
    end, total = -1, 0
    for left, right in sorted(intervals):
        if right > end:
            total += right - max(left, end)
            end = right
    return total


rows = []
for req in requests:
    lo, hi = req['start'], req['end']
    ks = [k for k in kernels if lo <= k['start'] < hi]
    cp = [r for r in copies if lo <= r['start'] < hi]
    gpu_union = union_ns(clipped(ks + cp, lo, hi))
    launches = [a for a in apis if lo <= a['start'] < hi and a['name'].startswith('cudaGraphLaunch')]
    rows.append({'request':req['name'], 'cpu_range_ms': (hi-lo)/1e6,
                 'gpu_kernel_count':len(ks), 'summed_kernel_ms':sum(k['end']-k['start'] for k in ks)/1e6,
                 'gpu_busy_union_ms':gpu_union/1e6, 'memcpy_count':len(cp),
                 # --cuda-trace-all-apis records versioned and unversioned
                 # aliases of one launch with the same correlation ID.
                 'cuda_graph_launches':len({(a['globalTid'],a['correlationId']) for a in launches}),
                 'gpu_streams':sorted({k['streamId'] for k in ks+cp})})
    assert ks and cp and launches, rows[-1]

native = Counter(r['name'] for r in nvtx if any(s in r['name'] for s in (
    'EngineCore', 'Scheduler.', 'UniProcExecutor.', 'AsyncOutputFuture.', 'GPUModelRunner.')))
adapters = [r for r in nvtx if r['name'].startswith('TEST_ONLY_FINAL_RESULT_ADAPTER/')]
assert any('step_with_batch_queue' in n for n in native), native
stack_tables = {t:con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
                for t in sorted(tables) if any(s in t for s in ('CALLCHAIN', 'PYTHON', 'SAMPL'))}
api_counts = Counter(r['name'] for r in apis)
coverage = {'requests': rows, 'total_kernels':len(kernels), 'total_memcopies':len(copies),
            'test_only_final_result_adapter_ranges':dict(Counter(r['name'] for r in adapters)),
            'native_nvtx_ranges':dict(native), 'stack_and_sampling_table_counts':stack_tables,
            'cuda_api_counts':dict(api_counts),
            'runtime_calls_with_callchains':sum(r.get('callchainId') is not None for r in apis),
            'stack_injection_diagnostics':[dict(r) for r in con.execute(
                "SELECT globalPid,text FROM DIAGNOSTIC_EVENT WHERE text LIKE '%Python%initialized successfully.%' OR text LIKE '%backtrace%initialized successfully.%'")],
            'note':'Profiled wall times include tracing overhead. CPU spans include waits; GPU busy time is a union, not a sum.'}
(HERE / 'nsys_trace_validation.json').write_text(json.dumps(coverage, indent=2)+'\n')

lanes = [
    ('Frontend request', '#78909c', requests),
    ('EngineCore.step_with_batch_queue', '#6a51a3', [r for r in nvtx if 'EngineCore.step_with_batch_queue' in r['name']]),
    ('Scheduler.schedule', '#4292c6', [r for r in nvtx if 'Scheduler.schedule' in r['name']]),
    ('Executor.execute_model', '#2171b5', [r for r in nvtx if 'UniProcExecutor.execute_model' in r['name']]),
    ('Executor.sample_tokens', '#41ab5d', [r for r in nvtx if 'UniProcExecutor.sample_tokens' in r['name']]),
    ('AsyncOutputFuture.result', '#ef6548', [r for r in nvtx if 'AsyncOutputFuture.result' in r['name']]),
    ('Scheduler.update_from_output', '#dd3497', [r for r in nvtx if 'Scheduler.update_from_output' in r['name']]),
    ('TEST ONLY: final result adapter', '#a63603', adapters),
]
for stream in sorted({k['streamId'] for k in kernels}):
    lanes.append((f'GPU kernels / stream {stream}', '#00a58e', [k for k in kernels if k['streamId']==stream]))
for kind in sorted({r['copyKind'] for r in copies}):
    lanes.append((f'GPU memcpy / {copy_types[kind]}', '#f2ad32', [r for r in copies if r['copyKind']==kind]))


def draw(ax, lo, hi):
    adapter_starts = [r['start'] for r in adapters if lo <= r['start'] < hi]
    if adapter_starts:
        ax.axvspan((min(adapter_starts)-lo)/1e6, (hi-lo)/1e6, color='#fdd0a2', alpha=.3)
    for y, (name, color, events) in enumerate(lanes):
        spans = [((left-lo)/1e6,(right-left)/1e6) for left,right in clipped(events,lo,hi)]
        ax.broken_barh(spans, (y-.32,.64), facecolors=color, edgecolors='none')
    ax.set_yticks(range(len(lanes)), [l[0] for l in lanes], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0,(hi-lo)/1e6)
    ax.set_xlabel('Time from request start (ms)', fontsize=9)
    ax.grid(axis='x', alpha=.18)
    ax.set_axisbelow(True)
    for spine in ('top','right'):
        ax.spines[spine].set_visible(False)


fig, axes = plt.subplots(5, 1, figsize=(17, 18), constrained_layout=True)
for ax, req, row in zip(axes, requests, rows):
    draw(ax,req['start'],req['end'])
    ax.set_title(f"{req['name']} | host span {row['cpu_range_ms']:.2f} ms | {row['gpu_kernel_count']} kernels | {row['cuda_graph_launches']} graph launches",loc='left',fontsize=11)
mode = json.loads((HERE/'nsys/run.json').read_text())['engine_config'].get('beam_execution_mode', 'v1')
fig.suptitle(f'OneRec-1.7B BF16 | offline {mode} / native async EngineCore | B1 / BW128 / 1024 input / SID3 | W10 + N5\nShaded tail: TEST ONLY result adapter. CPU spans include waits; full stack tracing adds overhead.',fontsize=14)
fig.savefig(HERE/'cpu_gpu_overlap.png', dpi=150)
fig.savefig(HERE/'cpu_gpu_overlap.svg')
plt.close(fig)
print(json.dumps(coverage,indent=2))
