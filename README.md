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

🚧 **Early stage — idea under validation.**

Currently working on M1: converting a single Linear layer end-to-end.

---

## Planned Flow

```
1. Parse    — extract weights + compute graph from any small HuggingFace model
2. Convert  — quantize (INT8) + normalize operators into a standard IR
3. Generate — emit Verilog modules + .mem weight files
4. Verify   — simulate with iverilog, compare against PyTorch reference
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
