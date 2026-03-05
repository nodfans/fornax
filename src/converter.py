"""
Fornax — Stage 2: Converter
---------------------------
Quantize float32 weights → INT8 and build Model IR.

Usage:
    from src.converter import ModelConverter
    converter = ModelConverter("./output")
    converter.convert()
    converter.save("./output")
"""

import os
import json
import numpy as np
from typing import Optional


class ModelConverter:
    def __init__(
        self,
        data_dir: str,
        target_dim: Optional[int] = None,
        strict_attention: bool = False,
        num_layers: Optional[int] = None,
        enable_embedding: bool = False,
        enable_final_norm: bool = False,
        enable_lm_head: bool = False,
        token_id: int = 0,
        vocab_limit: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Path to directory containing graph.json and weights/ folder.
        """
        self.data_dir = data_dir
        self.target_dim = target_dim
        self.strict_attention = strict_attention
        self.graph = {}
        self.weights = {}
        self.quantized_weights = {}  # { name: (int8_data, scale) }
        self.model_ir = {}
        self.model_dim = None
        self.strict_seq_len = 1
        self.num_layers = num_layers
        self.enable_embedding = bool(enable_embedding)
        self.enable_final_norm = bool(enable_final_norm)
        self.enable_lm_head = bool(enable_lm_head)
        self.token_id = int(token_id)
        self.vocab_limit = vocab_limit

        self._load_data()

    def _load_data(self):
        """Load graph.json and available .npy weights."""
        graph_path = os.path.join(self.data_dir, "graph.json")
        with open(graph_path, "r") as f:
            self.graph = json.load(f)

        weights_dir = os.path.join(self.data_dir, "weights")
        if os.path.exists(weights_dir):
            for f in os.listdir(weights_dir):
                if f.endswith(".npy"):
                    name = f[:-4]  # name with underscores
                    self.weights[name] = np.load(os.path.join(weights_dir, f))

    def convert(self):
        """Perform symmetric INT8 quantization and build IR."""
        print("[Converter] Quantizing weights to INT8...")
        for name, weight in self.weights.items():
            self.quantized_weights[name] = self._quantize_symmetric(weight)

        print("[Converter] Building Model IR...")
        self._build_ir()
        return self

    def save(self, output_dir: str):
        """Save q-weights and model_ir.json."""
        q_weights_dir = os.path.join(output_dir, "quantized_weights")
        os.makedirs(q_weights_dir, exist_ok=True)

        # Save quantized weights (.bin) and metadata
        weight_metadata = {}
        for name, (q_data, scale) in self.quantized_weights.items():
            path = os.path.join(q_weights_dir, f"{name}.bin")
            q_data.tofile(path)
            weight_metadata[name] = {
                "path": f"quantized_weights/{name}.bin",
                "scale": float(scale),
                "shape": list(q_data.shape),
                "dtype": "int8"
            }

        # Update IR with weight metadata
        self.model_ir["weight_metadata"] = weight_metadata

        ir_path = os.path.join(output_dir, "model_ir.json")
        with open(ir_path, "w") as f:
            json.dump(self.model_ir, f, indent=2)

        print(f"[Converter] Saved {len(self.quantized_weights)} quantized tensors.")
        print(f"[Converter] IR saved to {ir_path}")

    def _quantize_symmetric(self, weight: np.ndarray):
        """
        Symmetric INT8 quantization.
        scale = max(abs(weight)) / 127
        q = round(weight / scale)
        """
        max_val = np.max(np.abs(weight))
        if max_val == 0:
            return np.zeros(weight.shape, dtype=np.int8), 1.0
        
        scale = max_val / 127.0
        q_weight = np.round(weight / scale).astype(np.int8)
        return q_weight, scale

    def _calculate_ms_parameters(self, accumulator_scale: float, target_scale: float, shift: int = 16):
        """
        multiplier / (2^shift) = accumulator_scale / target_scale
        """
        ratio = accumulator_scale / target_scale
        multiplier = int(round(ratio * (2**shift)))
        return multiplier, shift

    def _build_ir(self):
        """Construct the IR representing the full Transformer Block with M3 precision."""
        self.model_dim = self._infer_model_dim()
        if self.target_dim is None:
            env_dim = os.environ.get("FORNAX_DIM")
            if env_dim:
                self.target_dim = int(env_dim)
            else:
                # Keep legacy fast path by default; can be overridden via arg/env.
                self.target_dim = min(128, self.model_dim)
        dim = max(1, int(self.target_dim))
        self.strict_seq_len = max(1, int(os.environ.get("FORNAX_STRICT_SEQ_LEN", "1")))
        total_layers = len(self.graph.get("layers", []))
        if self.num_layers is None:
            env_layers = os.environ.get("FORNAX_NUM_LAYERS")
            self.num_layers = int(env_layers) if env_layers else 1
        active_layers = max(1, min(int(self.num_layers), max(1, total_layers)))
        if not self.enable_embedding:
            self.enable_embedding = bool(int(os.environ.get("FORNAX_ENABLE_EMBED", "0")))
        if not self.enable_final_norm:
            self.enable_final_norm = bool(int(os.environ.get("FORNAX_ENABLE_FINAL_NORM", "0")))
        if not self.enable_lm_head:
            self.enable_lm_head = bool(int(os.environ.get("FORNAX_ENABLE_LM_HEAD", "0")))
        self.token_id = int(os.environ.get("FORNAX_TOKEN_ID", str(self.token_id)))
        if self.vocab_limit is None:
            env_vocab_limit = os.environ.get("FORNAX_VOCAB_LIMIT")
            if env_vocab_limit:
                self.vocab_limit = int(env_vocab_limit)
        
        ops = []
        target_int8_scale = 1.0 / 127.0
        layer_input = "block_input"

        # Optional M4 Task3 pre-block stage.
        if self.enable_embedding:
            embed_key = self._find_weight_key("model_embed_tokens_weight")
            if not embed_key:
                raise ValueError("Embedding is enabled but model_embed_tokens_weight is missing.")
            embed_vocab, embed_dim = self._resolve_embedding_dims(embed_key)
            if self.vocab_limit is not None:
                embed_vocab = max(1, min(embed_vocab, int(self.vocab_limit)))
            token_id = max(0, min(int(self.token_id), embed_vocab - 1))
            ops.append({
                "type": "embedding_lookup",
                "name": "tok_embedding",
                "input": "block_input",
                "weight_key": embed_key,
                "token_id": token_id,
                "vocab_size": embed_vocab,
                "dim": min(dim, embed_dim),
                "out_dtype": "int8"
            })
            layer_input = "tok_embedding"

        for lidx in range(active_layers):
            layer = self.graph["layers"][lidx]
            prefix = f"l{lidx}_"
            current_scale = target_int8_scale

            def n(base: str) -> str:
                return f"{prefix}{base}"

            # 1) input_layernorm
            ln1_data = next((op for op in layer["ops"] if "input_layernorm" in op["name"]), None)
            if ln1_data:
                w_name = ln1_data["weight"]
                _, w_scale = self.quantized_weights[w_name]
                acc_scale = current_scale * w_scale
                mult, shift = self._calculate_ms_parameters(acc_scale, target_int8_scale)
                ops.append({
                    "type": "layernorm",
                    "name": n("input_layernorm"),
                    "input": layer_input,
                    "dim": self._resolve_layernorm_dim(w_name, dim),
                    "weight_key": w_name,
                    "ms_multiplier": mult,
                    "ms_shift": shift
                })
                current_scale = target_int8_scale

            # 2) q/k/v projections
            prev_output = n("input_layernorm") if ln1_data else layer_input
            for proj in ["q_proj", "k_proj", "v_proj"]:
                op_data = next((op for op in layer["ops"] if proj in op["name"]), None)
                if not op_data:
                    continue
                w_name = op_data["weight"]
                _, w_scale = self.quantized_weights[w_name]
                acc_scale = current_scale * w_scale
                mult, shift = self._calculate_ms_parameters(acc_scale, target_int8_scale)
                out_features, in_features = self._resolve_linear_dims(w_name, dim, dim)
                ops.append({
                    "type": "linear",
                    "name": n(proj),
                    "input": prev_output,
                    "in_features": in_features,
                    "out_features": out_features,
                    "weight_key": w_name,
                    "ms_multiplier": mult,
                    "ms_shift": shift
                })

            # 3) attention interaction path
            q_op = next((o for o in ops if o["name"] == n("q_proj")), None)
            k_op = next((o for o in ops if o["name"] == n("k_proj")), None)
            v_op = next((o for o in ops if o["name"] == n("v_proj")), None)
            qk_dim = min(q_op["out_features"], k_op["out_features"]) if (q_op and k_op) else dim
            num_heads = int(self.graph.get("num_heads", 1))
            head_dim = max(1, int(self.graph.get("head_dim", 64)))
            seq_len = 1
            tensor_layout = "BHSd"
            scale_multiplier = int(round((1.0 / np.sqrt(head_dim)) * (2 ** 16)))
            context_dim = min(qk_dim, v_op["out_features"]) if v_op else qk_dim

            if self.strict_attention and self.strict_seq_len > 1:
                seq_len = self.strict_seq_len
                if qk_dim % seq_len != 0:
                    seq_len = 1
                req_heads = int(self.graph.get("num_kv_heads", num_heads))
                max_heads = max(1, qk_dim // seq_len)
                eff_heads = max(1, min(req_heads, max_heads))
                while eff_heads > 1 and (qk_dim % (seq_len * eff_heads) != 0):
                    eff_heads -= 1
                token_dim = max(1, qk_dim // (seq_len * eff_heads))
                strict_scale_multiplier = int(round((1.0 / np.sqrt(token_dim)) * (2 ** 16)))

                qk_op = {
                    "type": "qk_matmul_strict",
                    "name": n("attn_qk"),
                    "dim": qk_dim,
                    "seq_len": seq_len,
                    "num_heads": eff_heads,
                    "head_dim": token_dim,
                    "tensor_layout": tensor_layout,
                    "out_dtype": "int32"
                }
                self._attach_binary_inputs(qk_op, n("q_proj"), n("k_proj"))
                ops.append(qk_op)

                ops.append({
                    "type": "scale_scores",
                    "name": n("attn_scale"),
                    "input": n("attn_qk"),
                    "seq_len": seq_len,
                    "num_heads": eff_heads,
                    "multiplier": strict_scale_multiplier,
                    "shift": 16,
                    "out_dtype": "int8"
                })

                ops.append({
                    "type": "softmax_rows",
                    "name": n("attn_softmax"),
                    "input": n("attn_scale"),
                    "seq_len": seq_len,
                    "num_heads": eff_heads,
                    "out_dtype": "int8"
                })

                sv_op = {
                    "type": "sv_matmul_strict",
                    "name": n("attn_context"),
                    "dim": context_dim,
                    "seq_len": seq_len,
                    "num_heads": eff_heads,
                    "head_dim": token_dim,
                    "out_dtype": "int8"
                }
                self._attach_binary_inputs(sv_op, n("attn_softmax"), n("v_proj"))
                ops.append(sv_op)

            elif self.strict_attention:
                qk_op = {
                    "type": "qk_dot",
                    "name": n("attn_qk"),
                    "dim": qk_dim,
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "tensor_layout": tensor_layout,
                    "out_dtype": "int32"
                }
                self._attach_binary_inputs(qk_op, n("q_proj"), n("k_proj"))
                ops.append(qk_op)

                ops.append({
                    "type": "scale_scalar",
                    "name": n("attn_scale"),
                    "input": n("attn_qk"),
                    "multiplier": scale_multiplier,
                    "shift": 16,
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "tensor_layout": tensor_layout,
                    "out_dtype": "int32"
                })

                ops.append({
                    "type": "softmax_scalar",
                    "name": n("attn_softmax"),
                    "input": n("attn_scale"),
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "tensor_layout": tensor_layout,
                    "out_dtype": "int8"
                })

                sv_op = {
                    "type": "sv_matmul",
                    "name": n("attn_context"),
                    "dim": context_dim,
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "tensor_layout": tensor_layout,
                    "out_dtype": "int8"
                }
                self._attach_binary_inputs(sv_op, n("attn_softmax"), n("v_proj"))
                ops.append(sv_op)
            else:
                qk_op = {
                    "type": "matmul_qk",
                    "name": n("attn_qk"),
                    "dim": qk_dim,
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "tensor_layout": tensor_layout,
                    "out_dtype": "int8"
                }
                self._attach_binary_inputs(qk_op, n("q_proj"), n("k_proj"))
                ops.append(qk_op)

                ops.append({
                    "type": "scale",
                    "name": n("attn_scale"),
                    "input": n("attn_qk"),
                    "dim": qk_dim,
                    "multiplier": scale_multiplier,
                    "shift": 16,
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "tensor_layout": tensor_layout,
                    "out_dtype": "int8"
                })

                ops.append({
                    "type": "softmax",
                    "name": n("attn_softmax"),
                    "input": n("attn_scale"),
                    "dim": qk_dim,
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "tensor_layout": tensor_layout,
                    "out_dtype": "int8"
                })

                sv_op = {
                    "type": "matmul_sv",
                    "name": n("attn_context"),
                    "dim": context_dim,
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "tensor_layout": tensor_layout,
                    "out_dtype": "int8"
                }
                self._attach_binary_inputs(sv_op, n("attn_softmax"), n("v_proj"))
                ops.append(sv_op)

            # 4) o_proj
            op_data = next((op for op in layer["ops"] if "o_proj" in op["name"]), None)
            if op_data:
                w_name = op_data["weight"]
                _, w_scale = self.quantized_weights[w_name]
                acc_scale = target_int8_scale * w_scale
                mult, shift = self._calculate_ms_parameters(acc_scale, target_int8_scale)
                out_features, in_features = self._resolve_linear_dims(w_name, dim, context_dim)
                ops.append({
                    "type": "linear",
                    "name": n("o_proj"),
                    "input": n("attn_context"),
                    "in_features": in_features,
                    "out_features": out_features,
                    "weight_key": w_name,
                    "ms_multiplier": mult,
                    "ms_shift": shift
                })

            # 5) post-attn LN
            ln2_data = next((op for op in layer["ops"] if "post_attention_layernorm" in op["name"]), None)
            if ln2_data:
                w_name = ln2_data["weight"]
                _, w_scale = self.quantized_weights[w_name]
                acc_scale = target_int8_scale * w_scale
                mult, shift = self._calculate_ms_parameters(acc_scale, target_int8_scale)
                ops.append({
                    "type": "layernorm",
                    "name": n("post_attention_layernorm"),
                    "input": n("o_proj"),
                    "dim": self._resolve_layernorm_dim(w_name, dim),
                    "weight_key": w_name,
                    "ms_multiplier": mult,
                    "ms_shift": shift
                })

            # 6) MLP subset
            mlp_prev = n("post_attention_layernorm") if ln2_data else n("o_proj")
            for proj in ["gate_proj", "down_proj"]:
                op_data = next((op for op in layer["ops"] if proj in op["name"]), None)
                if not op_data:
                    continue
                w_name = op_data["weight"]
                _, w_scale = self.quantized_weights[w_name]
                acc_scale = target_int8_scale * w_scale
                mult, shift = self._calculate_ms_parameters(acc_scale, target_int8_scale)
                prev_dim = next((o["out_features"] for o in ops if o["name"] == mlp_prev and o["type"] == "linear"), dim)
                out_features, in_features = self._resolve_linear_dims(w_name, dim, prev_dim)
                ops.append({
                    "type": "linear",
                    "name": n(proj),
                    "input": mlp_prev,
                    "in_features": in_features,
                    "out_features": out_features,
                    "weight_key": w_name,
                    "ms_multiplier": mult,
                    "ms_shift": shift
                })
                mlp_prev = n(proj)

            layer_input = mlp_prev

        # Optional M4 Task3 post-block stage.
        if self.enable_final_norm:
            norm_key = self._find_weight_key("model_norm_weight")
            if not norm_key:
                raise ValueError("Final norm is enabled but model_norm_weight is missing.")
            _, norm_scale = self.quantized_weights[norm_key]
            acc_scale = target_int8_scale * norm_scale
            mult, shift = self._calculate_ms_parameters(acc_scale, target_int8_scale)
            ops.append({
                "type": "final_rmsnorm",
                "name": "final_rmsnorm",
                "input": layer_input,
                "dim": self._resolve_layernorm_dim(norm_key, dim),
                "weight_key": norm_key,
                "ms_multiplier": mult,
                "ms_shift": shift
            })
            layer_input = "final_rmsnorm"

        if self.enable_lm_head:
            lm_head_key = self._find_weight_key("lm_head_weight")
            if not lm_head_key:
                raise ValueError("LM head is enabled but lm_head_weight is missing.")
            _, lm_scale = self.quantized_weights[lm_head_key]
            # LM head output can be large; keep per-op scaling metadata for future RTL.
            acc_scale = target_int8_scale * lm_scale
            mult, shift = self._calculate_ms_parameters(acc_scale, target_int8_scale)
            requested_vocab = int(self.graph.get("vocab_size", 0)) or self.weights[lm_head_key].shape[0]
            if self.vocab_limit is not None:
                requested_vocab = min(requested_vocab, int(self.vocab_limit))
            out_features, in_features = self._resolve_linear_dims(lm_head_key, requested_vocab, dim)
            ops.append({
                "type": "lm_head",
                "name": "lm_head",
                "input": layer_input,
                "in_features": in_features,
                "out_features": out_features,
                "weight_key": lm_head_key,
                "ms_multiplier": mult,
                "ms_shift": shift
            })

        self.model_ir = {
            "version": "M4_TASK3_DRAFT" if (self.enable_embedding or self.enable_final_norm or self.enable_lm_head) else ("M3_STRICT_ATTN_DRAFT" if self.strict_attention else "M3"),
            "model_id": self.graph["model_id"],
            "ops": ops,
            "global_config": {
                "precision": "int8",
                "quantization": "symmetric",
                "target_scale": target_int8_scale,
                "model_dim": self.model_dim,
                "target_dim": dim,
                "strict_attention": bool(self.strict_attention),
                "num_layers": active_layers,
                "enable_embedding": bool(self.enable_embedding),
                "enable_final_norm": bool(self.enable_final_norm),
                "enable_lm_head": bool(self.enable_lm_head),
                "token_id": int(self.token_id),
                "vocab_limit": int(self.vocab_limit) if self.vocab_limit is not None else None,
            }
        }

    def _attach_binary_inputs(self, op: dict, input_a: str, input_b: str):
        if self.strict_attention:
            op["inputs"] = [input_a, input_b]
        else:
            op["input_a"] = input_a
            op["input_b"] = input_b

    def _infer_model_dim(self):
        if "hidden_size" in self.graph and self.graph["hidden_size"]:
            return int(self.graph["hidden_size"])
        layer0 = self.graph.get("layers", [{}])[0]
        for op in layer0.get("ops", []):
            if op.get("in_dim"):
                return int(op["in_dim"])
            if op.get("dim"):
                return int(op["dim"])
        # Fallback to first 2D weight's input dim.
        for arr in self.weights.values():
            if arr.ndim == 2:
                return int(arr.shape[1])
        return 128

    def _find_weight_key(self, base_name: str):
        if base_name in self.weights:
            return base_name
        for key in self.weights.keys():
            if key.endswith(base_name):
                return key
        return None

    def _resolve_embedding_dims(self, weight_key: str):
        shape = self.weights[weight_key].shape
        return int(shape[0]), int(shape[1])

    def _resolve_linear_dims(self, weight_key: str, requested_out: int, requested_in: int):
        shape = self.weights[weight_key].shape
        max_out, max_in = int(shape[0]), int(shape[1])
        out_features = min(int(requested_out), max_out)
        in_features = min(int(requested_in), max_in)
        return out_features, in_features

    def _resolve_layernorm_dim(self, weight_key: str, requested_dim: int):
        max_dim = int(self.weights[weight_key].shape[0])
        return min(int(requested_dim), max_dim)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./output")
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--strict-attn", action="store_true")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--enable-embed", action="store_true")
    parser.add_argument("--enable-final-norm", action="store_true")
    parser.add_argument("--enable-lm-head", action="store_true")
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--vocab-limit", type=int, default=None)
    args = parser.parse_args()
    
    con = ModelConverter(
        args.data_dir,
        target_dim=args.dim,
        strict_attention=args.strict_attn,
        num_layers=args.num_layers,
        enable_embedding=args.enable_embed,
        enable_final_norm=args.enable_final_norm,
        enable_lm_head=args.enable_lm_head,
        token_id=args.token_id,
        vocab_limit=args.vocab_limit,
    )
    con.convert()
    con.save(args.data_dir)
