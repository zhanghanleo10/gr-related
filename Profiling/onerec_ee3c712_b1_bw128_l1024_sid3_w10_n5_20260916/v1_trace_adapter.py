"""TEST ONLY: consume final persistent GPU records at V1's unimplemented A5 seam.

No scheduling, forward, candidate selection, KV, intermediate readback or
release behavior is replaced. Final D2H / sorting / packaging is diagnostic,
not a measurement of a native A5 implementation. Reconstruction follows
tests/kernels/test_beam_search_runner_cuda.py::_completed.
"""
from contextlib import contextmanager
from dataclasses import replace
import functools
import os
from types import SimpleNamespace


@contextmanager
def scope(label):
    import torch
    import nvtx
    with torch.profiler.record_function(label), nvtx.annotate(label, domain='test_adapter'):
        yield


def install():
    from vllm_gr.entrypoints.beam_search_v1.output_processor import (
        BeamSearchOutputProcessor, _raise_failure,
    )
    from vllm_gr.v1.worker.gpu_beam_stage_runner import AsyncGPUBeamOutput
    from vllm_gr.v1.engine.gr_async_scheduler import GR_WORKER_RESULT_ATTR
    from vllm_gr.v1.beam.batch_contracts import (
        BEAM_BATCH_PROTOCOL_VERSION, BeamBatchResult, BeamBatchOutput,
        BeamRequestOutput, BeamOutputSequence, validate_beam_batch_result,
    )
    if getattr(AsyncGPUBeamOutput, '_onerec_trace_adapter', False):
        return
    original = AsyncGPUBeamOutput._convert

    @functools.wraps(original)
    def convert(self, controls):
        output = original(self, controls)
        results = getattr(output, GR_WORKER_RESULT_ATTR)
        for entry in self.execution.entries:
            metadata = entry.metadata
            receipt = results[metadata.session_id]
            if not receipt.finished or receipt.terminal_result is not None:
                continue
            assert metadata.is_last_stage, 'This adapter only supports fixed SID3 completion'
            with scope('TEST_ONLY_FINAL_RESULT_ADAPTER/worker_D2H_sort_pack'):
                state = self.owner.beam.state
                index = entry.binding.state_row
                # Original get_output has already waited for the final control
                # snapshot. All model and Beam state writes precede that event.
                count = int(state.num_completed[index].cpu())
                assert count >= metadata.beam_params.beam_width
                names = ('token_ids', 'scores', 'parent_ids', 'lengths',
                         'finish_reasons', 'tie_break_indices')
                columns = [getattr(state, 'completed_' + n)[index, :count].cpu().tolist() for n in names]
                history_tokens = state.history_token_ids[index].cpu().tolist()
                history_parents = state.history_parent_ids[index].cpu().tolist()
                records = list(zip(*columns))
                assert all(r[3] == 3 and r[4] == 2 for r in records), 'Expected SID3 length completion'
                penalty = metadata.beam_params.length_penalty
                records.sort(key=lambda r: (-r[1] / r[3]**penalty, r[5]))
                sequences = []
                for token, score, parent, length, reason, tie in records[:metadata.output_options.num_return_sequences]:
                    tokens, parents = [token], [parent]
                    row = parent
                    for depth in range(length - 2, -1, -1):
                        tokens.append(history_tokens[depth][row])
                        row = history_parents[depth][row]
                        parents.append(row)
                    sequences.append(BeamOutputSequence(
                        tuple(reversed(tokens)), score, tuple(reversed(parents)), 'length', None,
                    ))
                terminal = BeamBatchResult(
                    BEAM_BATCH_PROTOCOL_VERSION, metadata.session_id,
                    BeamBatchOutput((BeamRequestOutput(
                        entry.binding.item_index, entry.binding.native_request_id, tuple(sequences),
                    ),)), None,
                )
                validate_beam_batch_result(terminal)
                results[metadata.session_id] = replace(receipt, terminal_result=terminal)
        return output

    def frontend(self, call, result):
        _raise_failure(result)
        validate_beam_batch_result(result)
        with scope('TEST_ONLY_FINAL_RESULT_ADAPTER/frontend_wrap'):
            begin = int(os.environ['ONEREC_TRACE_BEGIN_ID'])
            end = int(os.environ['ONEREC_TRACE_END_ID'])
            prefix = list(call.prompts[0]['prompt_token_ids']) + [begin]
            assert result.output is not None and len(result.output.items) == 1
            # Boundary tokens here are presentation framing, not GPU-generated SIDs.
            return [SimpleNamespace(sequences=[SimpleNamespace(
                tokens=prefix + list(sequence.token_ids) + [end],
                cum_logprob=sequence.cumulative_logprob,
            ) for sequence in result.output.items[0].sequences])]

    AsyncGPUBeamOutput._convert = convert
    AsyncGPUBeamOutput._onerec_trace_adapter = True
    BeamSearchOutputProcessor.ensure_available = classmethod(lambda cls: None)
    BeamSearchOutputProcessor.process_outputs = frontend
