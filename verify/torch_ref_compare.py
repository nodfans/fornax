#!/usr/bin/env python3
"""
Compare strict-attention integer path against a PyTorch reference attention.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import torch

ROWS_DIFF_MULT = 14


def _get_binary_inputs(op: dict[str, Any]) -> tuple[str, str]:
    if "inputs" in op and len(op["inputs"]) >= 2:
        return op["inputs"][0], op["inputs"][1]
    return op.get("input_a"), op.get("input_b")


def _exp_lut_rows(diff: int) -> int:
    table = [
        255, 239, 225, 211, 198, 186, 175, 164,
        154, 145, 136, 128, 120, 113, 106, 99,
        93, 88, 82, 77, 73, 68, 64, 60,
        56, 53, 50, 47, 44, 41, 39, 36,
        34, 32, 30, 28, 26, 25, 23, 22,
        20, 19, 18, 17, 16, 15, 14, 13,
        12, 11, 11, 10, 9, 9, 8, 8,
        7, 7, 6, 6, 5, 5, 5, 4,
        4, 4, 4, 3, 3, 3, 3, 3,
        2, 2, 2, 2, 2, 2, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    ]
    if diff < len(table):
        return table[diff]
    return 0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64).reshape(-1)
    bb = b.astype(np.float64).reshape(-1)
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    if den == 0:
        return 1.0
    return float(np.dot(aa, bb) / den)


def _load_ir(output_dir: str) -> dict[str, Any]:
    with open(os.path.join(output_dir, "model_ir.json"), "r") as f:
        return json.load(f)


def _load_qweight(output_dir: str, ir: dict[str, Any], key: str) -> np.ndarray:
    meta = ir["weight_metadata"][key]
    path = os.path.join(output_dir, meta["path"])
    return np.fromfile(path, dtype=np.int8).reshape(meta["shape"])


def _matvec_chunked(weights_i8: np.ndarray, vec_i32: np.ndarray, out_features: int, in_features: int, chunk: int = 1024) -> np.ndarray:
    out = np.zeros(out_features, dtype=np.int32)
    vec = vec_i32[:in_features].astype(np.int32)
    for start in range(0, out_features, chunk):
        end = min(start + chunk, out_features)
        w = weights_i8[start:end, :in_features].astype(np.int32, copy=False)
        out[start:end] = np.matmul(w, vec).astype(np.int32)
    return out


def _find_latest_name(keys: list[str], base: str) -> str:
    if base in keys:
        return base
    matches = [k for k in keys if k.endswith("_" + base)]
    if not matches:
        raise KeyError(f"Missing op/output for base name '{base}'")

    def layer_idx(name: str) -> int:
        if name.startswith("l"):
            p = name.find("_")
            if p > 1 and name[1:p].isdigit():
                return int(name[1:p])
        return -1

    return max(matches, key=layer_idx)


def _load_input_hex(output_dir: str) -> np.ndarray | None:
    path = os.path.join(output_dir, "testvectors", "input.hex")
    if not os.path.exists(path):
        return None
    vals: list[int] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            vals.append(int(s, 16) & 0xFF)
    if not vals:
        return None
    return np.array(vals, dtype=np.uint8).view(np.int8).astype(np.float32)


def _run_ir_attention(ir: dict[str, Any], output_dir: str, seed: int, input_vec: np.ndarray | None = None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    first_op = ir["ops"][0]
    dim = first_op.get("dim") or first_op.get("in_features")
    if input_vec is not None:
        x = input_vec.astype(np.float32)
    elif first_op.get("type") == "embedding_lookup":
        tok = int(first_op.get("token_id", 0)) & 0xFF
        x = np.array([tok], dtype=np.uint8).view(np.int8).astype(np.float32)
    else:
        x = rng.integers(-10, 10, size=(dim,), dtype=np.int8).astype(np.float32)
    results: dict[str, np.ndarray] = {"block_input": x}

    for i, op in enumerate(ir["ops"]):
        t = op["type"]
        if t == "embedding_lookup":
            w = _load_qweight(output_dir, ir, op["weight_key"])[: op["vocab_size"], : op["dim"]]
            token_id = int(op.get("token_id", 0))
            token_id = max(0, min(token_id, w.shape[0] - 1))
            results[op["name"]] = w[token_id].astype(np.int8).astype(np.float32)
        elif t == "layernorm" or t == "final_rmsnorm":
            w = _load_qweight(output_dir, ir, op["weight_key"]).reshape(-1)[: op["dim"]].astype(np.int32)
            inp = results[op["input"]].astype(np.int32)[: op["dim"]]
            acc = inp * w
            scaled = (acc.astype(np.int64) * np.int64(op["ms_multiplier"])) >> op["ms_shift"]
            results[op["name"]] = np.clip(scaled, -128, 127).astype(np.int8).astype(np.float32)
        elif t == "linear" or t == "lm_head":
            w = _load_qweight(output_dir, ir, op["weight_key"])[: op["out_features"], : op["in_features"]]
            inp_i32 = results[op["input"]].astype(np.int32)[: op["in_features"]]
            if t == "lm_head" and op["out_features"] >= 8192:
                acc = _matvec_chunked(w, inp_i32, op["out_features"], op["in_features"])
            else:
                acc = np.matmul(inp_i32.astype(np.float32), w.T.astype(np.float32)).astype(np.int32)
            if i < len(ir["ops"]) - 1:
                scaled = (acc.astype(np.int64) * np.int64(op["ms_multiplier"])) >> op["ms_shift"]
                results[op["name"]] = np.clip(scaled, -128, 127).astype(np.int8).astype(np.float32)
            else:
                results[op["name"]] = acc.astype(np.float32)
        elif t == "qk_dot":
            ia, ib = _get_binary_inputs(op)
            a = results[ia].astype(np.int32)
            b = results[ib].astype(np.int32)
            d = min(op["dim"], len(a), len(b))
            results[op["name"]] = np.array([int(np.sum(a[:d] * b[:d], dtype=np.int64))], dtype=np.int32).astype(np.float32)
        elif t == "qk_matmul_strict":
            ia, ib = _get_binary_inputs(op)
            a = results[ia].astype(np.int32)
            b = results[ib].astype(np.int32)
            h = int(op.get("num_heads", 1))
            s = int(op["seq_len"])
            d = int(op["head_dim"])
            total = h * s * d
            q = a[:total].reshape(h, s, d)
            k = b[:total].reshape(h, s, d)
            results[op["name"]] = np.matmul(q, np.transpose(k, (0, 2, 1))).astype(np.int32).reshape(-1).astype(np.float32)
        elif t == "scale_scalar":
            xv = int(results[op["input"]][0])
            y = (np.int64(xv) * np.int64(op["multiplier"])) >> op["shift"]
            results[op["name"]] = np.array([int(y)], dtype=np.int32).astype(np.float32)
        elif t == "scale_scores":
            x32 = results[op["input"]].astype(np.int32)
            h = int(op.get("num_heads", 1))
            s = int(op["seq_len"])
            total = h * s * s
            y = (x32[:total].astype(np.int64) * np.int64(op["multiplier"])) >> op["shift"]
            results[op["name"]] = np.clip(y, -32768, 32767).astype(np.int16).astype(np.float32)
        elif t == "softmax_scalar":
            results[op["name"]] = np.array([127], dtype=np.int8).astype(np.float32)
        elif t == "softmax_rows":
            x8 = results[op["input"]].astype(np.int32)
            h = int(op.get("num_heads", 1))
            s = int(op["seq_len"])
            mat = x8[: h * s * s].reshape(h, s, s)
            out = np.zeros_like(mat, dtype=np.int8)
            for hi in range(h):
                for r in range(s):
                    row = mat[hi, r]
                    m = int(np.max(row))
                    ex = np.array([_exp_lut_rows(int(m - v) * ROWS_DIFF_MULT) for v in row], dtype=np.int32)
                    row_sum = int(np.sum(ex))
                    denom = row_sum if row_sum > 0 else 1
                    out[hi, r] = np.clip((ex * 127) // denom, -128, 127).astype(np.int8)
            results[op["name"]] = out.reshape(-1).astype(np.float32)
        elif t == "sv_matmul":
            ia, ib = _get_binary_inputs(op)
            score = int(results[ia][0])
            v = results[ib].astype(np.int32)
            d = min(op["dim"], len(v))
            out = np.clip((score * v[:d]) >> 7, -128, 127).astype(np.int8)
            results[op["name"]] = out.astype(np.float32)
        elif t == "sv_matmul_strict":
            ia, ib = _get_binary_inputs(op)
            s8 = results[ia].astype(np.int32)
            v8 = results[ib].astype(np.int32)
            h = int(op.get("num_heads", 1))
            s = int(op["seq_len"])
            d = int(op["head_dim"])
            sm = s8[: h * s * s].reshape(h, s, s)
            vm = v8[: h * s * d].reshape(h, s, d)
            out = np.clip(np.matmul(sm, vm).astype(np.int32) >> 7, -128, 127).astype(np.int8)
            results[op["name"]] = out.reshape(-1).astype(np.float32)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--strict-only", action="store_true")
    args = parser.parse_args()

    ir = _load_ir(args.output)
    if args.strict_only and not ir.get("global_config", {}).get("strict_attention", False):
        raise SystemExit("IR is not strict_attention; rerun converter with --strict-attn.")

    input_vec = _load_input_hex(args.output)
    results = _run_ir_attention(ir, args.output, seed=args.seed, input_vec=input_vec)
    ops = {op["name"]: op for op in ir["ops"]}
    op_names = list(ops.keys())
    res_names = list(results.keys())
    q_name = _find_latest_name(res_names, "q_proj")
    k_name = _find_latest_name(res_names, "k_proj")
    v_name = _find_latest_name(res_names, "v_proj")
    qk_name = _find_latest_name(op_names, "attn_qk")
    softmax_name = _find_latest_name(res_names, "attn_softmax")
    ctx_name = _find_latest_name(res_names, "attn_context")

    # Determine strict attention shape from op metadata.
    qk_op = ops[qk_name]
    if qk_op.get("type") == "qk_dot":
        h = 1
        s = 1
        d = int(qk_op.get("dim", len(results[q_name])))
    else:
        h = int(qk_op.get("num_heads", 1))
        s = int(qk_op.get("seq_len", 1))
        d = int(qk_op.get("head_dim", max(1, len(results[q_name]) // max(1, h * s))))

    q_total = len(results[q_name])
    required = h * s * d
    if required > q_total:
        h = 1
        d = max(1, q_total // max(1, s))
        required = h * s * d

    q = torch.tensor(results[q_name][:required].astype(np.float32).reshape(h, s, d), dtype=torch.float32)
    k = torch.tensor(results[k_name][:required].astype(np.float32).reshape(h, s, d), dtype=torch.float32)
    v = torch.tensor(results[v_name][:required].astype(np.float32).reshape(h, s, d), dtype=torch.float32)

    logits_ref = torch.matmul(q, k.transpose(-1, -2))
    probs_ref = torch.softmax(logits_ref / np.sqrt(float(d)), dim=-1)
    probs_ref_q = torch.clamp(torch.round(probs_ref * 127.0), 0, 127).to(torch.int32)
    ctx_ref_q = torch.clamp(torch.matmul(probs_ref_q, v.to(torch.int32)) >> 7, -128, 127).to(torch.int32)

    # IR outputs
    logits_ir = results[qk_name].astype(np.int32).reshape(h, s, s)
    probs_ir = results[softmax_name].astype(np.int32).reshape(h, s, s)
    ctx_ir = results[ctx_name].astype(np.int32)[:required].reshape(h, s, d)

    logits_ref_np = logits_ref.detach().cpu().numpy().astype(np.int32)
    probs_ref_np = probs_ref_q.detach().cpu().numpy().astype(np.int32)
    ctx_ref_np = ctx_ref_q.detach().cpu().numpy().astype(np.int32)

    def report(name: str, a: np.ndarray, b: np.ndarray) -> None:
        delta = a.astype(np.int32) - b.astype(np.int32)
        mae = float(np.mean(np.abs(delta)))
        mxe = int(np.max(np.abs(delta)))
        cs = _cosine(a, b)
        print(f"[TORCH-REF] {name}: MAE={mae:.4f} MAX_ERR={mxe} COS={cs:.6f}")

    report("QK logits", logits_ir, logits_ref_np)
    report("Softmax(quantized)", probs_ir, probs_ref_np)
    report("Context(quantized)", ctx_ir, ctx_ref_np)

    if "lm_head" in ops:
        lm_name = _find_latest_name(list(ops.keys()), "lm_head")
        lm_op = ops[lm_name]
        in_name = lm_op["input"]
        in_vec = results[in_name].astype(np.int32)[: lm_op["in_features"]]
        w = _load_qweight(args.output, ir, lm_op["weight_key"])[: lm_op["out_features"], : lm_op["in_features"]]
        lm_ref = _matvec_chunked(w, in_vec, lm_op["out_features"], lm_op["in_features"], chunk=1024)
        lm_ir = results[lm_name].astype(np.int32)[: lm_op["out_features"]]
        report("LM Head logits", lm_ir, lm_ref)
        top1_ir = int(np.argmax(lm_ir))
        top1_ref = int(np.argmax(lm_ref))
        print(f"[TORCH-REF] LM Head top1: ir={top1_ir} ref={top1_ref} match={top1_ir == top1_ref}")


if __name__ == "__main__":
    main()
