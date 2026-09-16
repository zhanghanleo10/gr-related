"""Build the V1 canonical table from the exact real model and video catalog."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from tools.build_constraint_table import build_constraint_table

artifact, tokenizer, vocab = build_constraint_table(
    triples_path='/home/z00469465/OneRec/generated_data/video_constraint_triples.json',
    model='/home/z00469465/OneRec/OneRec-1.7B', revision=None,
    output_path=HERE/'video_constraint_table.safetensors', trust_remote_code=True,
)
config = {
    'schema_version': 1, 'attention_backend': 'CUSTOM',
    'beam': {'execution_mode': 'v1', 'graph_enabled': True, 'max_width':128,
             'max_decode_steps':3, 'worker_decision':True},
    'constraint_table': {'enabled': True, 'backend':'cuda', 'max_top_k':128,
                         'path':str(HERE/'video_constraint_table.safetensors'),
                         'format':'constraint_table_v1', 'artifact_digest':artifact,
                         'tokenizer_digest':tokenizer},
}
(HERE/'gr_config.json').write_text(json.dumps(config,indent=2)+'\n')
print(json.dumps({'artifact_digest':artifact, 'tokenizer_digest':tokenizer, 'vocab_size':vocab}))
