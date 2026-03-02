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

    def generate(self):
        """Generate Verilog modules and weight hex files."""
        print("[Generator] Generating Verilog for M2 Full Block...")
        
        # 1. Generate weight HEX files
        self._generate_weight_hex()

        # 2. Render each operation module
        rendered_modules = set()
        for op in self.model_ir["ops"]:
            # We render unique templates once, but they can be instanced multiple times
            if op["type"] == "linear" and "linear" not in rendered_modules:
                self._render_linear_template()
                rendered_modules.add("linear")
            elif op["type"] == "layernorm" and "layernorm" not in rendered_modules:
                self._render_layernorm_template()
                rendered_modules.add("layernorm")

        # 3. Render the Top module that connects everything
        self._render_top()

        print(f"[Generator] RTL and top.v emitted to {self.output_dir}/")

    def _generate_weight_hex(self):
        """Convert binary quantized weights to HEX string format for Verilog."""
        weight_metadata = self.model_ir.get("weight_metadata", {})
        
        for name, meta in weight_metadata.items():
            bin_path = os.path.join(self.data_dir, meta["path"])
            if not os.path.exists(bin_path):
                print(f"[Generator] Warning: Binary weight {bin_path} not found.")
                continue
            
            # Load as int8, view as uint8 for easy hex conversion
            weights = np.fromfile(bin_path, dtype=np.int8).view(np.uint8)
            
            hex_path = os.path.join(self.weights_out_dir, f"{name}.hex")
            with open(hex_path, "w") as f:
                for w in weights:
                    f.write(f"{w:02x}\n")
            
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

    def _render_top(self):
        """Render the top.v that connects all ops."""
        template = self.jinja_env.get_template("top.v.j2")
        rendered = template.render(ops=self.model_ir["ops"], weight_metadata=self.model_ir["weight_metadata"])
        with open(os.path.join(self.output_dir, "top.v"), "w") as f:
            f.write(rendered)
