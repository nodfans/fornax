# Strict Attention Spec (Draft)

Status: Draft for implementation alignment.
Scope: Define IR schema and RTL interfaces for true attention semantics in Fornax M3+.

## Goals

- Represent real attention dataflow: `QK^T -> scale -> softmax -> scores@V`.
- Support graph-shaped dependencies (fork/join), not only linear chains.
- Keep Stage-2/Stage-3 boundaries clear: converter defines graph, generator instantiates RTL.

## Non-Goals

- Full runtime accelerator architecture changes (DMA/DRAM/instruction set).
- Multi-token batching optimization in this draft.

## IR Schema Changes

## 1) Graph Inputs

Each op must support one of:

- `input` (single source), or
- `inputs` (multi-source list, length >= 2)

Recommendation:

- Keep `input` for unary ops.
- Use `inputs` for binary/n-ary ops (avoid ad-hoc `input_a/input_b` spread).

## 2) Op Types

- `linear` (existing): projects hidden vectors.
- `qk_matmul`: computes attention logits from `Q` and `K`.
- `scale`: fixed-point scale by `1/sqrt(head_dim)`.
- `softmax` (existing): normalize logits along key dimension.
- `sv_matmul`: multiply attention scores with `V`.

## 3) Shape Metadata (required)

Each op includes explicit shape fields so generator/testbench do not infer silently.

- `seq_len`: sequence length
- `num_heads`: attention heads
- `head_dim`: per-head dimension
- `tensor_layout`: one of `BHSd`, `BSHd`, etc.

For single-vector PoC (seq_len=1), still keep these fields for forward compatibility.

## 4) Quantization Metadata

Each op output should declare:

- `out_dtype` (e.g. `int8`, `int16`, `int32`)
- `scale_policy` (`ms`, `fixed_multiplier_shift`, `none`)
- scale params (`ms_multiplier/ms_shift` or `multiplier/shift`)

## 5) Versioning

- Set `model_ir.version` to `M3_STRICT_ATTN_DRAFT` while iterating.
- Move to `M3_STRICT_ATTN_V1` once schema is frozen.

## RTL Interface Spec

## 1) qk_matmul module

Inputs:

- `q_data`, `q_valid`
- `k_data`, `k_valid`
- shape params (`SEQ_LEN`, `NUM_HEADS`, `HEAD_DIM`)

Output:

- `logits_data` (`int32` recommended)
- `logits_valid`

Semantics:

- Compute dot-product over `head_dim` for each `(query_pos, key_pos, head)`.

## 2) scale module

Input:

- `in_data`, `in_valid` (typically logits)
- params: `MULTIPLIER`, `SHIFT`

Output:

- `out_data`, `out_valid`

Semantics:

- `out = saturate((in * MULTIPLIER) >>> SHIFT)`

## 3) softmax module

Input:

- logits stream with shape context

Output:

- normalized score stream

Semantics:

- normalize across key dimension per `(batch, head, query_pos)`.

## 4) sv_matmul module

Inputs:

- `score_data`, `score_valid`
- `v_data`, `v_valid`

Output:

- `context_data`, `context_valid`

Semantics:

- weighted sum across key dimension for each `(batch, head, query_pos)`.

## top.v Generation Rules

- Treat IR as DAG: instantiate by op list, wire by `input/inputs` names.
- No implicit chain assumptions.
- Validate that every dependency name exists before rendering.
- Emit compile-time error comments if schema is invalid.

## Verification Requirements

## 1) Reference Path

`verify/compare.py` should provide strict attention reference:

- `Q = x @ Wq`
- `K = x @ Wk`
- `V = x @ Wv`
- `logits = Q @ K^T`
- `scaled = logits / sqrt(head_dim)`
- `scores = softmax(scaled)`
- `context = scores @ V`

## 2) Intermediate Checks

Enable optional checks for:

- `qk_matmul` output
- `scale` output
- `softmax` output
- `sv_matmul` output

with tolerances by dtype.

## 3) CI Gate

- Keep existing lightweight regression.
- Add strict-attention regression case once strict IR path is implemented.

## Migration Plan

1. Converter emits both legacy and strict-attention IR behind a flag.
2. Generator supports both schemas.
3. Verify supports both schemas.
4. Switch default to strict schema after CI stability.

## Open Decisions

- Exact tensor layout convention (`BHSd` vs `BSHd`).
- Whether logits/scores use `int16` or `int32` internally.
- Streaming protocol for seq_len > 1 in current top-level interface.

