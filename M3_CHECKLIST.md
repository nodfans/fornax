# M3 Completion Checklist

Status: **Core M3 complete** (as of 2026-03-03).

## Completed

1. Full Layer-0 block validation at `DIM=128`
- End-to-end compare path (`expected.hex` vs `actual.hex`) is green.

2. True strict-attention semantics
- Implemented and validated path:
  - `QK^T -> scale -> softmax -> scores@V`
- Supports strict `seq_len=1` and strict `seq_len=4`.

3. Graph-capable IR for binary ops
- Strict schema uses `inputs: [a, b]`.
- Legacy schema remains supported for backward compatibility.

4. RTL + Python reference consistency
- `verify/compare.py` covers strict/legacy attention op variants.
- Regression fixture passes in legacy and strict modes.

5. PyTorch reference quality check
- Added optional reference comparison:
  - `FORNAX_TORCH_REF=1 ./run_m3_validation.sh 128`
- Current strict `seq_len=4` quality (latest):
  - `QK logits`: MAE=0, MAX_ERR=0, COS=1.0
  - `Softmax(quantized)`: MAE=3.75, MAX_ERR=11, COS=0.991971
  - `Context(quantized)`: MAE=0, MAX_ERR=0, COS=1.0

## CI Gate

Current required regression jobs:
- `regression-legacy`
- `regression-strict` (`seq_len=1`)
- `regression-strict-seq4` (`FORNAX_STRICT_SEQ_LEN=4`)

## Remaining (Post-M3 / M4+)

1. Scale from `DIM=128` to model-native dimensions in default validation flow.
2. Multi-layer chaining and full-model path (embedding/output head).
3. GQA-faithful head mapping (`num_heads` vs `num_kv_heads`) beyond current strict flattened path.
4. Hardware cost/latency tracking (LUT/FF/BRAM/timing) for strict attention modules.
5. Stabilize simulator behavior by avoiding parallel runs into the same `output/` directory.
