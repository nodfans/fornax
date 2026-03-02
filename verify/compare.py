import numpy as np
import os
import json

def generate_test_vectors(output_dir="./output"):
    print("[Verify] Generating test vectors for M2 Block...")
    ir_path = os.path.join(output_dir, "model_ir.json")
    with open(ir_path, "r") as f:
        ir = json.load(f)
    
    # Generate random input (1 x DIM)
    # Typically LLM hidden size is ir['ops'][0]['dim']
    dim = ir["ops"][0].get("dim") or ir["ops"][0].get("in_features")
    input_vec = np.random.randint(-10, 10, size=(dim,), dtype=np.int8)
    
    current_val = input_vec.astype(np.float32)
    
    # Run through the IR chain in Python
    for op in ir["ops"]:
        weight_meta = ir["weight_metadata"][op["weight_key"]]
        bin_path = os.path.join(output_dir, weight_meta["path"])
        weights = np.fromfile(bin_path, dtype=np.int8)
        
        if op["type"] == "layernorm":
            # Simplified M2 LN: just multiply by weight (scale)
            # weights is (DIM,)
            current_val = current_val * weights.astype(np.float32)
            # Re-quantize to INT8 for next stage
            current_val = np.clip(current_val, -128, 127).astype(np.int8).astype(np.float32)
        elif op["type"] == "linear":
            weights = weights.reshape(op["out_features"], op["in_features"])
            # Linear: x * W^T
            current_val = np.matmul(current_val, weights.T.astype(np.float32))
            # In RTL we take bits [15:8], which is a division by 256
            current_val = (current_val / 256.0).astype(np.float32)
            current_val = np.clip(current_val, -128, 127).astype(np.int8).astype(np.float32)

    expected_out = current_val # Final output features
    
    # Save files
    testvec_dir = os.path.join(output_dir, "testvectors")
    os.makedirs(testvec_dir, exist_ok=True)
    with open(os.path.join(testvec_dir, "input.hex"), "w") as f:
        for v in input_vec.view(np.uint8):
            f.write(f"{v:02x}\n")
    with open(os.path.join(testvec_dir, "expected.hex"), "w") as f:
        # Match the width of the last op in top.v (typically 32-bit linear if last is linear)
        # For simplify M2, let's assume we output the 32-bit value 
        for v in expected_out.astype(np.int32).view(np.uint32):
            f.write(f"{v:08X}\n")

    print(f"[Verify] Test vectors saved.")

def check_results(output_dir="./output"):
    print("\n[Verify] Comparing Simulation Results...")
    testvec_dir = os.path.join(output_dir, "testvectors")
    
    expected_path = os.path.join(testvec_dir, "expected.hex")
    actual_path = os.path.join(testvec_dir, "actual.hex")
    
    if not os.path.exists(actual_path):
        print(f"❌ actual.hex not found at {actual_path}. Did you run the simulation?")
        return
    
    with open(expected_path, "r") as f:
        expected = [int(line.strip(), 16) for line in f if line.strip()]
        # Convert back to signed int32 if needed (hex is uint32 view)
        expected = [e if e < 0x80000000 else e - 0x100000000 for e in expected]
        
    with open(actual_path, "r") as f:
        actual = [int(line.strip(), 16) for line in f if line.strip()]
        actual = [a if a < 0x80000000 else a - 0x100000000 for a in actual]
        
    match_count = 0
    for i in range(min(len(expected), len(actual))):
        if expected[i] == actual[i]:
            match_count += 1
        else:
            print(f"  [MISMATCH] Index {i}: Expected {expected[i]}, Got {actual[i]}")
            
    if match_count == len(expected) and len(expected) == len(actual):
        print(f"✅ PASS: All {match_count} outputs match perfectly!")
    else:
        print(f"❌ FAIL: {match_count}/{len(expected)} matches.")

def verify_m1(output_dir="./output"):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", action="store_true", help="Force generate new test vectors")
    parser.add_argument("--check", action="store_true", help="Only check existing simulation results")
    args, _ = parser.parse_known_args()

    print("--- [Stage 4: Verify] ---")
    
    # Step 1: Check files
    ir_path = os.path.join(output_dir, "model_ir.json")
    if not os.path.exists(ir_path):
        print("❌ model_ir.json not found. Run convert.py first.")
        return

    testvec_dir = os.path.join(output_dir, "testvectors")
    input_hex = os.path.join(testvec_dir, "input.hex")

    if args.check:
        check_results(output_dir)
        return

    # Step 2: Generate test vectors if forced or missing
    if args.gen or not os.path.exists(input_hex):
        generate_test_vectors(output_dir)
    else:
        print("[Verify] Using existing test vectors.")
    
    # Step 3: Check for simulation output
    check_results(output_dir)

if __name__ == "__main__":
    verify_m1()
