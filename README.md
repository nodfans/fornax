# 🔥 Fornax

> Compile small open-source LLMs into FPGA-ready Verilog + weight files.

---

## Idea

Most tools let you *run* a model on hardware.
Fornax does something different — it *translates* the model into hardware.

```
HuggingFace Model  →  [ Fornax ]  →  Verilog + Weights
```

Inspired by what [Taalas](https://taalas.com) is doing with custom ASICs:
instead of loading weights at runtime, **the weights become the circuit**.

Fornax brings this idea to the open-source world, targeting FPGA developers first.

---

## Status

✅ **M4 is complete for Qwen2-0.5B (INT8).**

What is validated today:
- End-to-end graph: `embedding -> 24 layers -> final_rmsnorm -> lm_head`
- Native hidden size: `DIM=896`
- Strict attention path: `QK^T -> scale -> softmax -> scores@V`
- RTL exact match:
  - token e2e with `vocab4k`: PASS
  - token e2e with `vocab16k`: PASS
- PyTorch reference:
  - `QK/Softmax/Context/LM Head logits`: `MAE=0`, `COS=1`
  - `top1` token: match
  - full vocab reference (`151936`): PASS

---

## Planned Flow

```
1. Parse    — extract weights + compute graph from any small HuggingFace model
2. Convert  — quantize (INT8) + normalize operators into a standard IR
3. Generate — emit Verilog modules + .mem weight files
4. Verify   — simulate with iverilog, compare against PyTorch reference
```

## Quick Start

Block / layer-stack RTL:

```bash
./run_m3_validation.sh 128
```

Strict attention (`seq_len=1`):

```bash
FORNAX_STRICT_ATTN=1 ./run_m3_validation.sh 128
```

Strict attention (`seq_len=4`):

```bash
FORNAX_STRICT_ATTN=1 FORNAX_STRICT_SEQ_LEN=4 ./run_m3_validation.sh 128
```

Multi-layer chain (example: 2 layers):

```bash
FORNAX_NUM_LAYERS=2 FORNAX_STRICT_ATTN=1 ./run_m3_validation.sh 128
```

Strict attention + PyTorch reference metrics:

```bash
FORNAX_STRICT_ATTN=1 FORNAX_STRICT_SEQ_LEN=4 FORNAX_TORCH_REF=1 ./run_m3_validation.sh 128
```

Token e2e RTL (staged vocab):

```bash
FORNAX_NUM_LAYERS=24 FORNAX_STRICT_ATTN=1 FORNAX_STRICT_SEQ_LEN=1 \
FORNAX_ENABLE_EMBED=1 FORNAX_ENABLE_FINAL_NORM=1 FORNAX_ENABLE_LM_HEAD=1 \
FORNAX_TOKEN_ID=42 FORNAX_VOCAB_LIMIT=4096 FORNAX_TORCH_REF=1 \
./run_m3_validation.sh 896
```

```bash
FORNAX_NUM_LAYERS=24 FORNAX_STRICT_ATTN=1 FORNAX_STRICT_SEQ_LEN=1 \
FORNAX_ENABLE_EMBED=1 FORNAX_ENABLE_FINAL_NORM=1 FORNAX_ENABLE_LM_HEAD=1 \
FORNAX_TOKEN_ID=42 FORNAX_VOCAB_LIMIT=16384 FORNAX_TORCH_REF=1 \
./run_m3_validation.sh 896
```

Full-vocab reference compare (no RTL run):

```bash
FORNAX_NUM_LAYERS=24 FORNAX_STRICT_ATTN=1 FORNAX_TOKEN_ID=42 \
./run_m4_full_vocab_ref.sh 896
```

CI regression:

```bash
python verify/run_regression.py
python verify/run_regression.py --strict-attn
FORNAX_STRICT_SEQ_LEN=4 python verify/run_regression.py --strict-attn
```

Notes:
- `run_m3_validation.sh` writes to `output/` by default.
- Use different `FORNAX_OUTPUT_DIR` for parallel or overnight runs.

```bash
FORNAX_OUTPUT_DIR=./artifacts/run_a FORNAX_NUM_LAYERS=24 FORNAX_STRICT_ATTN=1 ./run_m3_validation.sh 128
FORNAX_OUTPUT_DIR=./artifacts/run_b FORNAX_NUM_LAYERS=2 FORNAX_STRICT_ATTN=1 FORNAX_STRICT_SEQ_LEN=4 ./run_m3_validation.sh 896
```

If parsed model data is not in `./output`:

```bash
FORNAX_MODEL_DATA_DIR=/path/to/model_dump FORNAX_OUTPUT_DIR=./artifacts/run_a ./run_m3_validation.sh 128
```

Debug mode:

```bash
# Default: no TRACE/LN-DEBUG/VCD (faster)
# Enable debug traces and wave dump only when needed:
FORNAX_DEBUG=1 ./run_m3_validation.sh 128
```

---

## Target Models

Small open-source models under 2B parameters:
Qwen2-0.5B, Llama 3.2 1B, Phi-3 Mini, Gemma 2B.

---

## Why

- No open-source tool converts Transformers to Verilog
- TVM is too general
- FINN only supports CNNs
- Taalas is closed source

Fornax fills that gap.

---

*Fornax — Latin for "furnace". Where models are recast into silicon.*
