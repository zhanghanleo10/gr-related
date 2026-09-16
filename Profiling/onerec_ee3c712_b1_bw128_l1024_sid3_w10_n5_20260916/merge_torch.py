"""Merge frontend and worker Kineto exports without dropping any events."""
import gzip
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
paths = sorted((HERE / 'torch').glob('*.pt.trace.json'))
assert len(paths) == 2, paths
traces = [json.loads(path.read_text()) for path in paths]
bases = [int(t.get('baseTimeNanoseconds', 0)) for t in traces]
assert all(bases), bases
base = min(bases)
merged = {'schemaVersion': 1, 'displayTimeUnit': 'ms', 'baseTimeNanoseconds': base,
          'traceEvents': [], 'stackFrames': {},
          'source_files': [p.name for p in paths],
          'merge_method': 'timestamp shift from baseTimeNanoseconds; no event filtering'}
for key in ('deviceProperties', 'with_stack', 'record_shapes', 'profile_memory',
            'cuda_driver_version', 'cuda_runtime_version', 'cupti_version'):
    for trace in traces:
        if key in trace:
            merged[key] = trace[key]
reports = []
flow_ids = {}
for index, (path, trace, trace_base) in enumerate(zip(paths, traces, bases)):
    events = trace['traceEvents']
    for event in events:
        if 'ts' in event:
            event['ts'] += (trace_base - base) / 1000
        # Scope flow and stack IDs to each profiler, preserving its correlations.
        if event.get('ph') in ('s', 't', 'f') and 'id' in event:
            key = (index, str(event['id']))
            event['id'] = flow_ids.setdefault(key, len(flow_ids) + 1)
        if 'sf' in event:
            event['sf'] = f'{index}:{event["sf"]}'
    for key, frame in trace.get('stackFrames', {}).items():
        if 'parent' in frame:
            frame['parent'] = f'{index}:{frame["parent"]}'
        merged['stackFrames'][f'{index}:{key}'] = frame
    merged['traceEvents'].extend(events)
    native = [e for e in events if 'vllm/' in str(e.get('name', ''))]
    reports.append({'file': path.name, 'event_count': len(events),
                    'categories': dict(Counter(e.get('cat', '') for e in events)),
                    'native_vllm_python_events': len(native),
                    'native_vllm_examples': list(dict.fromkeys(e['name'] for e in native))[:35]})
categories = Counter(e.get('cat', '') for e in merged['traceEvents'])
requests = [e for e in merged['traceEvents'] if e.get('ph') == 'X' and e.get('name', '').startswith('onerec_request_')]
assert len(requests) == 5, len(requests)
assert categories['kernel'] > 0 and categories['python_function'] > 0, categories
assert sum(r['native_vllm_python_events'] for r in reports) > 0
target = HERE / 'full_cpu_gpu.pt.trace.json'
with target.open('w') as out:
    json.dump(merged, out, separators=(',', ':'))
with target.open('rb') as source, gzip.open(str(target) + '.gz', 'wb', compresslevel=6) as out:
    import shutil
    shutil.copyfileobj(source, out)
summary = {'files': reports, 'total_events': len(merged['traceEvents']),
           'categories': dict(categories), 'requests': [{'name':e['name'], 'cpu_range_ms':e['dur']/1000} for e in requests]}
(HERE / 'torch_trace_validation.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
