# Fornax Context (Current)

## Project Purpose
Fornax compiles small open-source Transformer models into Verilog plus weight data.
Scope is compiler/generation and simulation-based verification, not runtime accelerator design.

## Current Milestone Status
- M1: Done
- M2: Done
- M3: Done
- M4: Done (Qwen2-0.5B INT8 path)

## What M4 Delivers
- Model: `Qwen/Qwen2-0.5B`
- Precision: `INT8`
- End-to-end graph:
  `embedding -> 24 transformer layers -> final_rmsnorm -> lm_head`
- Native hidden size: `896`
- Strict attention path:
  `QK^T -> scale -> softmax -> scores@V`

## Validation Snapshot
- RTL exact compare:
  - 24L + DIM896 + token e2e + vocab4k: PASS
  - 24L + DIM896 + token e2e + vocab16k: PASS
- PyTorch reference:
  - QK/Softmax/Context/LM Head logits: `MAE=0`, `COS=1`
  - top1 token: match
  - full-vocab reference path (`151936`) match

## Operational Notes
- Full-vocab RTL remains very slow due to serial matmul behavior.
- Recommended acceptance split:
  - staged vocab RTL (e.g. 4k/16k),
  - full-vocab PyTorch/reference compare.
- Debug trace and VCD are off by default; enable with `FORNAX_DEBUG=1`.

## Next Focus
- M5: Multi-model support.
- Performance improvements for lm_head RTL (top-k/candidate flow, parallelism).
