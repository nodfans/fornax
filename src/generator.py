"""
Fornax — Stage 3: Generator
---------------------------
Emit Verilog code and weight hex files from Model IR.

Usage:
    from src.generator import VerilogGenerator
    gen = VerilogGenerator("./output")
    gen.generate()
"""

import os
import json
import numpy as np
from jinja2 import Environment, FileSystemLoader


class VerilogGenerator:
    def __init__(self, data_dir: str, template_dir: str = "./templates"):
        """
        Args:
            data_dir: Directory containing model_ir.json and quantized_weights/
            template_dir: Directory containing Jinja2 Verilog templates
        """
        self.data_dir = data_dir
        self.template_dir = template_dir
        self.output_dir = os.path.join(data_dir, "rtl")
        self.weights_out_dir = os.path.join(data_dir, "weights_hex")
        self.model_ir = {}

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.weights_out_dir, exist_ok=True)

        self._load_ir()
        self.jinja_env = Environment(loader=FileSystemLoader(self.template_dir))

    def _load_ir(self):
        ir_path = os.path.join(self.data_dir, "model_ir.json")
        with open(ir_path, "r") as f:
            self.model_ir = json.load(f)

    def _validate_ir_graph(self):
        """Validate op dependencies before rendering."""
        ops = self.model_ir.get("ops", [])
        known = {"block_input"}
        for op in ops:
            deps = []
            if "input" in op:
                deps.append(op["input"])
            if "inputs" in op:
                deps.extend(op["inputs"])
            else:
                if "input_a" in op:
                    deps.append(op["input_a"])
                if "input_b" in op:
                    deps.append(op["input_b"])
            for d in deps:
                if d not in known:
                    raise ValueError(f"IR dependency '{d}' for op '{op.get('name')}' is undefined.")
            known.add(op.get("name"))

    def generate(self):
        """Generate Verilog modules and weight hex files."""
        print("[Generator] Generating Verilog for M2 Full Block...")
        self._validate_ir_graph()
        
        # 1. Generate weight HEX files
        self._generate_weight_hex()

        # 2. Render each operation module
        rendered_modules = set()
        for op in self.model_ir["ops"]:
            # We render unique templates once, but they can be instanced multiple times
            if op["type"] == "linear" and "linear" not in rendered_modules:
                self._render_linear_template()
                rendered_modules.add("linear")
            elif op["type"] == "lm_head" and "linear" not in rendered_modules:
                self._render_linear_template()
                rendered_modules.add("linear")
            elif op["type"] == "layernorm" and "layernorm" not in rendered_modules:
                self._render_layernorm_template()
                rendered_modules.add("layernorm")
            elif op["type"] == "final_rmsnorm" and "layernorm" not in rendered_modules:
                self._render_layernorm_template()
                rendered_modules.add("layernorm")
            elif op["type"] == "embedding_lookup" and "embedding_lookup" not in rendered_modules:
                self._render_embedding_lookup_template()
                rendered_modules.add("embedding_lookup")
            elif op["type"] == "softmax" and "softmax" not in rendered_modules:
                self._render_softmax_template()
                rendered_modules.add("softmax")
            elif op["type"] == "matmul_qk" and "matmul_qk" not in rendered_modules:
                self._render_matmul_qk_template()
                rendered_modules.add("matmul_qk")
            elif op["type"] == "scale" and "scale" not in rendered_modules:
                self._render_scale_template()
                rendered_modules.add("scale")
            elif op["type"] == "matmul_sv" and "matmul_sv" not in rendered_modules:
                self._render_matmul_sv_template()
                rendered_modules.add("matmul_sv")
            elif op["type"] == "qk_dot" and "qk_dot" not in rendered_modules:
                self._render_qk_dot_template()
                rendered_modules.add("qk_dot")
            elif op["type"] == "scale_scalar" and "scale_scalar" not in rendered_modules:
                self._render_scale_scalar_template()
                rendered_modules.add("scale_scalar")
            elif op["type"] == "softmax_scalar" and "softmax_scalar" not in rendered_modules:
                self._render_softmax_scalar_template()
                rendered_modules.add("softmax_scalar")
            elif op["type"] == "sv_matmul" and "sv_matmul" not in rendered_modules:
                self._render_sv_matmul_template()
                rendered_modules.add("sv_matmul")
            elif op["type"] == "qk_matmul_strict" and "qk_matmul_strict" not in rendered_modules:
                self._render_qk_matmul_strict_template()
                rendered_modules.add("qk_matmul_strict")
            elif op["type"] == "scale_scores" and "scale_scores" not in rendered_modules:
                self._render_scale_scores_template()
                rendered_modules.add("scale_scores")
            elif op["type"] == "softmax_rows" and "softmax_rows" not in rendered_modules:
                self._render_softmax_rows_template()
                rendered_modules.add("softmax_rows")
            elif op["type"] == "sv_matmul_strict" and "sv_matmul_strict" not in rendered_modules:
                self._render_sv_matmul_strict_template()
                rendered_modules.add("sv_matmul_strict")

        # 3. Render the Top module that connects everything
        self._render_top()

        print(f"[Generator] RTL and top.v emitted to {self.output_dir}/")

    def _generate_weight_hex(self):
        """Convert binary quantized weights to HEX string format for Verilog."""
        weight_metadata = self.model_ir.get("weight_metadata", {})
        
        # OPTIMIZATION: Only generate hex for weights used in this specific IR
        used_weights = {op["weight_key"] for op in self.model_ir["ops"] if "weight_key" in op}
        
        for name in used_weights:
            if name not in weight_metadata:
                print(f"[Generator] Warning: Metadata for {name} not found in IR.")
                continue
            meta = weight_metadata[name]
            bin_path = os.path.join(self.data_dir, meta["path"])
            if not os.path.exists(bin_path):
                print(f"[Generator] Warning: Binary weight {bin_path} not found.")
                continue
            
            # Load and slice weights if they are larger than IR dimensions
            weights = np.fromfile(bin_path, dtype=np.int8).reshape(meta["shape"])
            
            # Find the op that uses this weight to get dimensions
            op = next((o for o in self.model_ir["ops"] if o.get("weight_key") == name), None)
            if op:
                if op["type"] == "linear":
                    weights = weights[:op["out_features"], :op["in_features"]]
                elif op["type"] == "lm_head":
                    weights = weights[:op["out_features"], :op["in_features"]]
                elif op["type"] == "embedding_lookup":
                    weights = weights[:op["vocab_size"], :op["dim"]]
                elif op["type"] == "layernorm":
                    weights = weights[:op["dim"]]
                elif op["type"] == "final_rmsnorm":
                    weights = weights[:op["dim"]]
            
            # Flatten once sliced and view as uint8 for hex
            weights = weights.flatten().view(np.uint8)
            
            hex_path = os.path.join(self.weights_out_dir, f"{name}.hex")
            # OPTIMIZATION: Process in chunks and use join to reduce write calls
            chunk_size = 65536
            with open(hex_path, "w") as f:
                for i in range(0, len(weights), chunk_size):
                    chunk = weights[i:i+chunk_size]
                    f.write("\n".join([f"{w:02x}" for w in chunk]) + "\n")
            
            meta["hex_path"] = f"../weights_hex/{name}.hex"

    def _render_linear_template(self):
        """Render the generic matmul.v template."""
        # Note: M2 uses a generic template that is parameterized by instantiations in top.v
        template = self.jinja_env.get_template("matmul.v.j2")
        # We render it with default parameters or just the logic
        # For M2, we'll keep the template generic and pass params in top.v
        # But we need to make sure the template doesn't have hardcoded values
        rendered = template.render(
            in_features="IN_FEATURES",
            out_features="OUT_FEATURES",
            weight_file="WEIGHT_FILE"
        )
        with open(os.path.join(self.output_dir, "matmul.v"), "w") as f:
            f.write(rendered)

    def _render_layernorm_template(self):
        """Render the layernorm.v template."""
        template = self.jinja_env.get_template("layernorm.v.j2")
        rendered = template.render()
        with open(os.path.join(self.output_dir, "layernorm.v"), "w") as f:
            f.write(rendered)

    def _render_embedding_lookup_template(self):
        """Render embedding lookup module."""
        template = self.jinja_env.get_template("embedding_lookup.v.j2")
        rendered = template.render()
        with open(os.path.join(self.output_dir, "embedding_lookup.v"), "w") as f:
            f.write(rendered)

    def _render_softmax_template(self):
        """Render the softmax.v template."""
        template = self.jinja_env.get_template("softmax.v.j2")
        rendered = template.render()
        with open(os.path.join(self.output_dir, "softmax.v"), "w") as f:
            f.write(rendered)

    def _render_matmul_qk_template(self):
        """Render QK interaction module."""
        template = self.jinja_env.get_template("matmul_qk.v.j2")
        rendered = template.render()
        with open(os.path.join(self.output_dir, "matmul_qk.v"), "w") as f:
            f.write(rendered)

    def _render_scale_template(self):
        """Render scaling module."""
        template = self.jinja_env.get_template("scale.v.j2")
        rendered = template.render()
        with open(os.path.join(self.output_dir, "scale.v"), "w") as f:
            f.write(rendered)

    def _render_matmul_sv_template(self):
        """Render score/value interaction module."""
        template = self.jinja_env.get_template("matmul_sv.v.j2")
        rendered = template.render()
        with open(os.path.join(self.output_dir, "matmul_sv.v"), "w") as f:
            f.write(rendered)

    def _render_qk_dot_template(self):
        template = self.jinja_env.get_template("qk_dot.v.j2")
        rendered = template.render()
        with open(os.path.join(self.output_dir, "qk_dot.v"), "w") as f:
            f.write(rendered)

    def _render_scale_scalar_template(self):
        template = self.jinja_env.get_template("scale_scalar.v.j2")
        rendered = template.render()
        with open(os.path.join(self.output_dir, "scale_scalar.v"), "w") as f:
            f.write(rendered)

    def _render_softmax_scalar_template(self):
        template = self.jinja_env.get_template("softmax_scalar.v.j2")
        rendered = template.render()
        with open(os.path.join(self.output_dir, "softmax_scalar.v"), "w") as f:
            f.write(rendered)

    def _render_sv_matmul_template(self):
        template = self.jinja_env.get_template("sv_matmul.v.j2")
        rendered = template.render()
        with open(os.path.join(self.output_dir, "sv_matmul.v"), "w") as f:
            f.write(rendered)

    def _render_qk_matmul_strict_template(self):
        template = self.jinja_env.get_template("qk_matmul_strict.v.j2")
        rendered = template.render()
        with open(os.path.join(self.output_dir, "qk_matmul_strict.v"), "w") as f:
            f.write(rendered)

    def _render_scale_scores_template(self):
        template = self.jinja_env.get_template("scale_scores.v.j2")
        rendered = template.render()
        with open(os.path.join(self.output_dir, "scale_scores.v"), "w") as f:
            f.write(rendered)

    def _render_softmax_rows_template(self):
        template = self.jinja_env.get_template("softmax_rows.v.j2")
        rendered = template.render()
        with open(os.path.join(self.output_dir, "softmax_rows.v"), "w") as f:
            f.write(rendered)

    def _render_sv_matmul_strict_template(self):
        template = self.jinja_env.get_template("sv_matmul_strict.v.j2")
        rendered = template.render()
        with open(os.path.join(self.output_dir, "sv_matmul_strict.v"), "w") as f:
            f.write(rendered)

    def _render_top(self):
        """Render the top.v that connects all ops."""
        template = self.jinja_env.get_template("top.v.j2")
        # Create a dict for easy lookup in Jinja
        ops_dict = {op["name"]: op for op in self.model_ir["ops"]}
        rendered = template.render(
            ops=self.model_ir["ops"], 
            weight_metadata=self.model_ir["weight_metadata"],
            ops_dict=ops_dict
        )
        with open(os.path.join(self.output_dir, "top.v"), "w") as f:
            f.write(rendered)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./output")
    args = parser.parse_args()
    
    gen = VerilogGenerator(args.output)
    gen.generate()
