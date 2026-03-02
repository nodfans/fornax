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


class ModelConverter:
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to directory containing graph.json and weights/ folder.
        """
        self.data_dir = data_dir
        self.graph = {}
        self.weights = {}
        self.quantized_weights = {}  # { name: (int8_data, scale) }
        self.model_ir = {}

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

    def _build_ir(self):
        """Construct the IR representing the full Transformer Block (M2)."""
        layer0 = self.graph["layers"][0]
        
        ops = []
        
        # 1. input_layernorm (Reads from block input)
        ln1 = next((op for op in layer0["ops"] if "input_layernorm" in op["name"]), None)
        if ln1:
            ops.append({
                "type": "layernorm",
                "name": "input_layernorm",
                "input": "block_input",
                "dim": ln1["dim"],
                "weight_key": ln1["weight"]
            })

        # 2. Attention Projections (Q, K, V all read from ln1)
        # Note: In M2 we treat them as independent modules to prove multi-instance works
        prev_output = "input_layernorm" if ln1 else "block_input"
        
        for proj in ["q_proj", "k_proj", "v_proj"]:
            op_data = next((op for op in layer0["ops"] if proj in op["name"]), None)
            if op_data:
                ops.append({
                    "type": "linear",
                    "name": proj,
                    "input": prev_output,
                    "in_features": op_data["in_dim"],
                    "out_features": op_data["out_dim"],
                    "weight_key": op_data["weight"]
                })

        # o_proj (Simplified: in M2 we just chain it after q_proj for PoC)
        op_data = next((op for op in layer0["ops"] if "o_proj" in op["name"]), None)
        if op_data:
            ops.append({
                "type": "linear",
                "name": "o_proj",
                "input": "q_proj", 
                "in_features": op_data["in_dim"],
                "out_features": op_data["out_dim"],
                "weight_key": op_data["weight"]
            })

        # 3. post_attention_layernorm (Reads from o_proj)
        ln2 = next((op for op in layer0["ops"] if "post_attention_layernorm" in op["name"]), None)
        if ln2:
            ops.append({
                "type": "layernorm",
                "name": "post_attention_layernorm",
                "input": "o_proj",
                "dim": ln2["dim"],
                "weight_key": ln2["weight"]
            })

        # 4. MLP (gate_proj, then down_proj)
        # We simplify the MLP path for M2 to a serial chain: gate -> down
        mlp_prev = "post_attention_layernorm" if ln2 else "o_proj"
        for proj in ["gate_proj", "down_proj"]:
            op_data = next((op for op in layer0["ops"] if proj in op["name"]), None)
            if op_data:
                ops.append({
                    "type": "linear",
                    "name": proj,
                    "input": mlp_prev,
                    "in_features": op_data["in_dim"],
                    "out_features": op_data["out_dim"],
                    "weight_key": op_data["weight"]
                })
                mlp_prev = proj

        self.model_ir = {
            "version": "M2",
            "model_id": self.graph["model_id"],
            "ops": ops,
            "global_config": {
                "precision": "int8",
                "quantization": "symmetric"
            }
        }
