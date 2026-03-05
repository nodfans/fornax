import argparse
import sys
import os

from src.parser import ModelParser
from src.converter import ModelConverter
from src.generator import VerilogGenerator

def main():
    parser = argparse.ArgumentParser(description="Fornax: LLM to Verilog Compiler")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B", help="HuggingFace model ID")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--dim", type=int, default=None, help="Target feature dimension for generated IR/RTL")
    parser.add_argument("--strict-attn", action="store_true", help="Emit strict-attention IR schema (inputs list for binary ops)")
    args = parser.parse_args()
    
    print("🚀 Fornax Compiler Starting...")
    
    # Stage 1: Parse
    print("\n--- [Stage 1: Parse] ---")
    stg1 = ModelParser(args.model)
    stg1.parse()
    stg1.save(args.output)
    
    # Stage 2: Convert
    print("\n--- [Stage 2: Convert] ---")
    stg2 = ModelConverter(args.output, target_dim=args.dim, strict_attention=args.strict_attn)
    stg2.convert()
    stg2.save(args.output)

    # Stage 3: Generate
    print("\n--- [Stage 3: Generate] ---")
    stg3 = VerilogGenerator(args.output)
    stg3.generate()
    
    print("\n✅ M1 End-to-End Pipeline Complete.")
    
if __name__ == "__main__":
    main()
