# Changelog

All notable changes to this project will be documented in this file.

## [0.0.1] - 2026-03-05

### Added
- End-to-end INT8 token path for Qwen2-0.5B:
  `embedding -> 24 layers -> final_rmsnorm -> lm_head`.
- New IR op coverage for Task3 components:
  `embedding_lookup`, `final_rmsnorm`, `lm_head`.
- Full-vocab reference validation entrypoint:
  `run_m4_full_vocab_ref.sh`.
- `embedding_lookup` RTL template and top-level integration.

### Changed
- Multi-layer IR generation and naming for chained layer validation.
- `run_m3_validation.sh`:
  supports Task3 env flags (`FORNAX_ENABLE_*`, `FORNAX_TOKEN_ID`, `FORNAX_VOCAB_LIMIT`),
  auto-derives testbench `IN_LEN/OUT_LEN`, and supports debug compile flag.
- `verify/tb_top.v`:
  generalized input/output lengths and fixed 64-bit timeout arithmetic.
- `verify/compare.py` and `verify/torch_ref_compare.py`:
  support Task3 ops and lm_head end-to-end comparison.
- Simulation debug behavior:
  TRACE/LN-DEBUG/VCD disabled by default; enabled with `FORNAX_DEBUG=1`.
- CI regression workflow:
  isolated per-job artifact/output directories to avoid path conflicts in parallel jobs.

### Verified
- RTL exact compare PASS:
  - 24 layers, DIM=896, token e2e, vocab4k
  - 24 layers, DIM=896, token e2e, vocab16k
- PyTorch reference PASS:
  - QK/Softmax/Context/LM Head logits (`MAE=0`, `COS=1`)
  - top1 token match
  - full-vocab lm_head reference path (151936) match
