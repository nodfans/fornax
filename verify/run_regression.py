#!/usr/bin/env python3
"""
Run a deterministic M3/M2 validation regression on synthetic fixture data.

This avoids downloading large HuggingFace models and is suitable for CI.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
import sys
import argparse

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.converter import ModelConverter
from src.generator import VerilogGenerator
from verify.compare import check_results, generate_test_vectors


DIM = 128
SEED = 42


def _build_fixture(output_dir: Path, seed: int = SEED) -> None:
    rng = np.random.default_rng(seed)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    graph = {
        "model_id": "fixture/m3-regression",
        "layers": [
            {
                "layer_idx": 0,
                "ops": [
                    {"name": "input_layernorm", "type": "layernorm", "weight": "model_layers_0_input_layernorm_weight"},
                    {"name": "q_proj", "type": "linear", "weight": "model_layers_0_self_attn_q_proj_weight"},
                    {"name": "k_proj", "type": "linear", "weight": "model_layers_0_self_attn_k_proj_weight"},
                    {"name": "v_proj", "type": "linear", "weight": "model_layers_0_self_attn_v_proj_weight"},
                    {"name": "o_proj", "type": "linear", "weight": "model_layers_0_self_attn_o_proj_weight"},
                    {"name": "post_attention_layernorm", "type": "layernorm", "weight": "model_layers_0_post_attention_layernorm_weight"},
                    {"name": "gate_proj", "type": "linear", "weight": "model_layers_0_mlp_gate_proj_weight"},
                    {"name": "down_proj", "type": "linear", "weight": "model_layers_0_mlp_down_proj_weight"},
                ],
            }
        ],
    }
    (output_dir / "graph.json").write_text(json.dumps(graph, indent=2))

    def save_weight(name: str, shape: tuple[int, ...]) -> None:
        data = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        np.save(weights_dir / f"{name}.npy", data)

    save_weight("model_layers_0_input_layernorm_weight", (DIM,))
    save_weight("model_layers_0_post_attention_layernorm_weight", (DIM,))
    save_weight("model_layers_0_self_attn_q_proj_weight", (DIM, DIM))
    save_weight("model_layers_0_self_attn_k_proj_weight", (DIM, DIM))
    save_weight("model_layers_0_self_attn_v_proj_weight", (DIM, DIM))
    save_weight("model_layers_0_self_attn_o_proj_weight", (DIM, DIM))
    save_weight("model_layers_0_mlp_gate_proj_weight", (DIM, DIM))
    save_weight("model_layers_0_mlp_down_proj_weight", (DIM, DIM))


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Optional directory to keep regression artifacts for CI debugging.",
    )
    parser.add_argument(
        "--strict-attn",
        action="store_true",
        help="Enable strict-attention IR schema (binary ops use inputs list).",
    )
    args = parser.parse_args()

    repo_root = REPO_ROOT
    tb_src = repo_root / "verify" / "tb_top.v"

    if args.artifact_dir:
        out_dir = Path(args.artifact_dir).resolve() / "output"
        if out_dir.exists():
            for p in sorted(out_dir.rglob("*"), reverse=True):
                if p.is_file():
                    try:
                        p.unlink()
                    except FileNotFoundError:
                        pass
            for p in sorted(out_dir.rglob("*"), reverse=True):
                if p.is_dir():
                    try:
                        p.rmdir()
                    except FileNotFoundError:
                        pass
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_regression(repo_root, tb_src, out_dir, strict_attention=args.strict_attn)
        print(f"[REGRESSION] PASS. Artifacts in {out_dir}")
        return

    with tempfile.TemporaryDirectory(prefix="fornax-reg-") as td:
        out_dir = Path(td) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_regression(repo_root, tb_src, out_dir, strict_attention=args.strict_attn)
        print(f"[REGRESSION] PASS. Artifacts in {out_dir}")


def _run_regression(repo_root: Path, tb_src: Path, out_dir: Path, strict_attention: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    _build_fixture(out_dir, seed=SEED)

    converter = ModelConverter(str(out_dir), strict_attention=strict_attention)
    converter.convert()
    converter.save(str(out_dir))

    generator = VerilogGenerator(str(out_dir), template_dir=str(repo_root / "templates"))
    generator.generate()

    tb_dir = out_dir / "testbench"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_rendered = (
        tb_src.read_text()
        .replace("__DIM__", str(DIM))
        .replace("__LAYERS__", "1")
        .replace("__IN_LEN__", str(DIM))
        .replace("__OUT_LEN__", str(DIM))
    )
    (tb_dir / "tb_top.v").write_text(tb_rendered)

    generate_test_vectors(output_dir=str(out_dir), dim=DIM)

    rtl_files = sorted((out_dir / "rtl").glob("*.v"))
    compile_cmd = ["iverilog", "-o", "sim_check"] + [str(p) for p in rtl_files] + ["tb_top.v"]
    _run(compile_cmd, cwd=tb_dir)
    with open(tb_dir / "sim_check.log", "w") as f:
        subprocess.run(["vvp", "sim_check"], cwd=str(tb_dir), check=True, stdout=f, stderr=subprocess.STDOUT)

    ok = check_results(output_dir=str(out_dir), dim=DIM)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
