"""Validate both completed V1 captures and package the reviewable artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
outputs = []
for mode in ('nsys','torch'):
    run = json.loads((HERE/mode/'run.json').read_text())
    assert run['engine_config']['vllm_gr_config']['beam']['execution_mode'] == 'v1'
    assert run['contract']['warmups'] == len(run['warmups']) == 10
    assert run['contract']['profiled_requests'] == len(run['samples']) == 5
    assert len(run['validation']) == 5 and run['all_outputs_identical']
    assert run['test_only_final_result_adapter']
    assert 'CAPTURE_COMPLETE' in (HERE/mode/'capture.log').read_text()
    outputs.extend(row['sha256'] for row in run['validation'])
assert len(set(outputs)) == 1, 'Two profiler runs produced different outputs'
tracked_diff = subprocess.check_output(['git','diff','--stat'],cwd=HERE.parents[1],text=True)
assert not tracked_diff, tracked_diff

files = [p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.sh','.md','.json','.png','.svg','.safetensors')]
files.append(HERE/'full_cpu_gpu.pt.trace.json.gz')
files.extend(HERE/'nsys'/name for name in ('run.json','capture.log','full_trace.nsys-rep','full_trace.sqlite'))
files.extend(p for p in (HERE/'torch').iterdir() if p.is_file())
files = sorted(set(files))
manifest=[]
for path in files:
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    manifest.append(f'{digest}  {path.relative_to(HERE)}')
(HERE/'MANIFEST.sha256').write_text('\n'.join(manifest)+'\n')
files.append(HERE/'MANIFEST.sha256')
archive=HERE.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
    for path in files:
        tar.add(path,arcname=str(Path(HERE.name)/path.relative_to(HERE)))
print(json.dumps({'archive':str(archive),'archive_bytes':archive.stat().st_size,
                  'files':len(files),'all_10_profiled_outputs_sha256':outputs[0],
                  'tracked_source_diff':tracked_diff},indent=2))
