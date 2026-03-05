import numpy as np
import os
import json
import re

ROWS_DIFF_MULT = 14

def _get_binary_inputs(op):
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

def generate_test_vectors(output_dir="./output", dim=896):
    print("[Verify] Generating test vectors for M2 Block...")
    ir_path = os.path.join(output_dir, "model_ir.json")
    with open(ir_path, "r") as f:
        ir = json.load(f)
    
    # Generate input vector for current IR entry op.
    # Default block mode feeds DIM int8 values; embedding mode only needs one trigger token.
    first_op = ir["ops"][0]
    dim = first_op.get("dim") or first_op.get("in_features")
    if first_op.get("type") == "embedding_lookup":
        tok = int(first_op.get("token_id", 0)) & 0xFF
        input_vec = np.array([tok], dtype=np.uint8).view(np.int8)
    else:
        input_vec = np.random.randint(-10, 10, size=(dim,), dtype=np.int8)
    
    current_val = input_vec.astype(np.float32)
    
    # Run through the IR chain in Python
    results = {"block_input": current_val}
    
    for i, op in enumerate(ir["ops"]):
        if op["type"] == "embedding_lookup":
            weight_meta = ir["weight_metadata"][op["weight_key"]]
            bin_path = os.path.join(output_dir, weight_meta["path"])
            shape = weight_meta["shape"]
            weights = np.fromfile(bin_path, dtype=np.int8).reshape(shape)
            token_id = int(op.get("token_id", 0))
            token_id = max(0, min(token_id, shape[0] - 1))
            out = weights[token_id, :op["dim"]].astype(np.int8)
            results[op["name"]] = out.astype(np.float32)
            first_5 = [f"{int(x) & 0xFF:02x}" for x in out[:5]]
            print(f"[PY-TRACE] Op:{op['name']} first_5: {', '.join(first_5)}")

        elif op["type"] == "layernorm" or op["type"] == "final_rmsnorm":
            op_input = results[op["input"]]
            weight_meta = ir["weight_metadata"][op["weight_key"]]
            bin_path = os.path.join(output_dir, weight_meta["path"])
            weights = np.fromfile(bin_path, dtype=np.int8).astype(np.int32)
            # Slice weights to match IR dimension
            weights = weights[:op["dim"]]
            
            # Simulated M-S Scaling
            acc = op_input.astype(np.int32) * weights
            # (acc * mult) >> shift
            scaled = (acc.astype(np.int64) * np.int64(op["ms_multiplier"])) >> op["ms_shift"]
            res_bits = np.clip(scaled, -128, 127).astype(np.int8)
            results[op["name"]] = res_bits.astype(np.float32)
            first_5 = [f"{int(x) & 0xFF:02x}" for x in res_bits[:5]]
            print(f"[PY-TRACE] Op:{op['name']} first_5: {', '.join(first_5)}")

        elif op["type"] == "linear" or op["type"] == "lm_head":
            op_input = results[op["input"]]
            weight_meta = ir["weight_metadata"][op["weight_key"]]
            bin_path = os.path.join(output_dir, weight_meta["path"])
            # Load according to metadata shape, then slice
            shape = weight_meta["shape"]
            raw_weights = np.fromfile(bin_path, dtype=np.int8).reshape(shape)
            weights = raw_weights[:op["out_features"], :op["in_features"]]
            
            res = np.matmul(op_input, weights.T.astype(np.float32)).astype(np.int32)
            
            # M3: Every linear op is scaled using M-S before next stage
            # Use the multiplier and shift calculated in converter.py
            scaled = (res.astype(np.int64) * np.int64(op["ms_multiplier"])) >> op["ms_shift"]
            res_bits = np.clip(scaled, -128, 127).astype(np.int8)
            
            if i < len(ir["ops"]) - 1:
                results[op["name"]] = res_bits.astype(np.float32)
                print(f"[PY-TRACE] Op:{op['name']} scaled_out: {int(res_bits[0]) & 0xFF:02x}")
            else:
                # Last op: keep 32-bit (Testbench currently expects 32-bit for the very last assignment)
                results[op["name"]] = res
                print(f"[PY-TRACE] Op:{op['name']} raw_out: {int(res[0]) & 0xFFFFFFFF:08x}")

        elif op["type"] == "matmul_qk":
            in_a, in_b = _get_binary_inputs(op)
            a = results[in_a].astype(np.int32)
            b = results[in_b].astype(np.int32)
            dim = min(op["dim"], len(a), len(b))
            prod = a[:dim] * b[:dim]
            shifted = prod >> 7
            out = np.clip(shifted, -128, 127).astype(np.int8)
            results[op["name"]] = out.astype(np.float32)
            print(f"[PY-TRACE] Op:{op['name']} first_out: {int(out[0]) & 0xFF:02x}")

        elif op["type"] == "qk_dot":
            in_a, in_b = _get_binary_inputs(op)
            a = results[in_a].astype(np.int32)
            b = results[in_b].astype(np.int32)
            dim = min(op["dim"], len(a), len(b))
            dot = int(np.sum(a[:dim] * b[:dim], dtype=np.int64))
            results[op["name"]] = np.array([dot], dtype=np.int32).astype(np.float32)
            print(f"[PY-TRACE] Op:{op['name']} dot: {dot}")

        elif op["type"] == "qk_matmul_strict":
            in_a, in_b = _get_binary_inputs(op)
            a = results[in_a].astype(np.int32)
            b = results[in_b].astype(np.int32)
            seq_len = int(op["seq_len"])
            num_heads = int(op.get("num_heads", 1))
            head_dim = int(op["head_dim"])
            total = num_heads * seq_len * head_dim
            q = a[:total].reshape(num_heads, seq_len, head_dim)
            k = b[:total].reshape(num_heads, seq_len, head_dim)
            logits = np.matmul(q, np.transpose(k, (0, 2, 1))).astype(np.int32)
            flat = logits.reshape(-1)
            results[op["name"]] = flat.astype(np.float32)
            print(f"[PY-TRACE] Op:{op['name']} first_out: {int(flat[0]) & 0xFFFFFFFF:08x}")

        elif op["type"] == "scale":
            x = results[op["input"]].astype(np.int32)
            dim = min(op["dim"], len(x))
            scaled = (x[:dim].astype(np.int64) * np.int64(op["multiplier"])) >> op["shift"]
            out = np.clip(scaled, -128, 127).astype(np.int8)
            results[op["name"]] = out.astype(np.float32)
            print(f"[PY-TRACE] Op:{op['name']} first_out: {int(out[0]) & 0xFF:02x}")

        elif op["type"] == "scale_scores":
            x = results[op["input"]].astype(np.int32)
            seq_len = int(op["seq_len"])
            num_heads = int(op.get("num_heads", 1))
            total = num_heads * seq_len * seq_len
            scaled = (x[:total].astype(np.int64) * np.int64(op["multiplier"])) >> op["shift"]
            out = np.clip(scaled, -32768, 32767).astype(np.int16)
            results[op["name"]] = out.astype(np.float32)
            print(f"[PY-TRACE] Op:{op['name']} first_out: {int(out[0]) & 0xFFFF:04x}")

        elif op["type"] == "scale_scalar":
            x = int(results[op["input"]][0])
            y = (np.int64(x) * np.int64(op["multiplier"])) >> op["shift"]
            results[op["name"]] = np.array([int(y)], dtype=np.int32).astype(np.float32)
            print(f"[PY-TRACE] Op:{op['name']} scalar: {int(y)}")

        elif op["type"] == "matmul_sv":
            in_a, in_b = _get_binary_inputs(op)
            a = results[in_a].astype(np.int32)
            b = results[in_b].astype(np.int32)
            dim = min(op["dim"], len(a), len(b))
            prod = a[:dim] * b[:dim]
            shifted = prod >> 7
            out = np.clip(shifted, -128, 127).astype(np.int8)
            results[op["name"]] = out.astype(np.float32)
            print(f"[PY-TRACE] Op:{op['name']} first_out: {int(out[0]) & 0xFF:02x}")

        elif op["type"] == "sv_matmul":
            in_a, in_b = _get_binary_inputs(op)
            score = int(results[in_a][0]) if len(results[in_a]) > 0 else 0
            b = results[in_b].astype(np.int32)
            dim = min(op["dim"], len(b))
            prod = score * b[:dim]
            shifted = prod >> 7
            out = np.clip(shifted, -128, 127).astype(np.int8)
            results[op["name"]] = out.astype(np.float32)
            print(f"[PY-TRACE] Op:{op['name']} first_out: {int(out[0]) & 0xFF:02x}")

        elif op["type"] == "softmax":
            op_input = results[op["input"]]
            # REPLICATE Verilog Softmax (3-pass LUT approach)
            x = op_input.astype(np.int16)
            max_val = np.max(x)
            
            # Simple Python LUT (matches softmax.v.j2's base points)
            # Match Verilog LUT: exp(-x/16.0) * 255
            lut = [255, 239, 225, 211, 198, 186, 175, 164, 154, 145, 136, 128, 120, 113, 106, 99, 93, 88, 82, 77, 73, 68, 64, 60, 56, 53, 50, 47, 44, 41, 39, 36, 34, 32, 30, 28, 26, 25, 23, 22, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 11, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
            # PASS 2: Sum Exp
            sum_exp = 0
            exps = []
            for val in x:
                diff = int(max_val - val)
                e = lut[diff] if diff < 128 else 0
                exps.append(e)
                sum_exp += e
            
            # PASS 3: Normalize (Matching PASS3 logic)
            norm_factor = (int(sum_exp) >> 8) + 1
            out = []
            for e in exps:
                q_out = (int(e) * 127) // norm_factor
                out.append(np.clip(q_out, -128, 127))
            
            results[op["name"]] = np.array(out, dtype=np.int8).astype(np.float32)
            print(f"[PY-TRACE] Op:{op['name']} first_out: {int(out[0]) & 0xFF:02x}")

        elif op["type"] == "softmax_scalar":
            results[op["name"]] = np.array([127], dtype=np.int8).astype(np.float32)
            print(f"[PY-TRACE] Op:{op['name']} scalar_out: 7f")

        elif op["type"] == "softmax_rows":
            x = results[op["input"]].astype(np.int32)
            seq_len = int(op["seq_len"])
            num_heads = int(op.get("num_heads", 1))
            total = num_heads * seq_len * seq_len
            mat = x[:total].reshape(num_heads, seq_len, seq_len)
            out_mat = np.zeros_like(mat, dtype=np.int8)
            for h in range(num_heads):
                for r in range(seq_len):
                    row = mat[h, r]
                    row_max = int(np.max(row))
                    exp_vals = np.array([_exp_lut_rows(int(row_max - v) * ROWS_DIFF_MULT) for v in row], dtype=np.int32)
                    row_sum = int(np.sum(exp_vals))
                    denom = row_sum if row_sum > 0 else 1
                    norm = (exp_vals * 127) // denom
                    out_mat[h, r] = np.clip(norm, -128, 127).astype(np.int8)
            out = out_mat.reshape(-1)
            results[op["name"]] = out.astype(np.float32)
            print(f"[PY-TRACE] Op:{op['name']} first_out: {int(out[0]) & 0xFF:02x}")

        elif op["type"] == "sv_matmul_strict":
            in_a, in_b = _get_binary_inputs(op)
            scores = results[in_a].astype(np.int32)
            v = results[in_b].astype(np.int32)
            seq_len = int(op["seq_len"])
            num_heads = int(op.get("num_heads", 1))
            head_dim = int(op["head_dim"])
            s_total = num_heads * seq_len * seq_len
            v_total = num_heads * seq_len * head_dim
            score_mat = scores[:s_total].reshape(num_heads, seq_len, seq_len)
            v_mat = v[:v_total].reshape(num_heads, seq_len, head_dim)
            ctx = np.matmul(score_mat, v_mat).astype(np.int32)
            shifted = ctx >> 7
            out = np.clip(shifted, -128, 127).astype(np.int8).reshape(-1)
            results[op["name"]] = out.astype(np.float32)
            print(f"[PY-TRACE] Op:{op['name']} first_out: {int(out[0]) & 0xFF:02x}")

    # Final output is the result of the last op
    expected_out = results[ir["ops"][-1]["name"]]
    
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

def check_results(output_dir="./output", dim=896):
    print("\n[Verify] Comparing Simulation Results...")
    expected_path = os.path.join(output_dir, "testvectors/expected.hex")
    actual_path = os.path.join(output_dir, "testvectors/actual.hex")
    
    if not os.path.exists(actual_path):
        print(f"❌ FAIL: Simulation output {actual_path} not found.")
        return False

    if not os.path.exists(expected_path):
        print(f"❌ FAIL: Expected output {expected_path} not found.")
        return False

    def hex_to_signed_int32(h):
        val = int(h, 16)
        if val & (1 << 31): # Check if the sign bit (31st bit for 32-bit) is set
            val -= (1 << 32) # Convert to negative
        return val

    expected = []
    invalid_expected = 0
    with open(expected_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if re.fullmatch(r"[0-9a-fA-F]+", s):
                expected.append(hex_to_signed_int32(s))
            else:
                invalid_expected += 1
    
    actual = []
    invalid_actual = 0
    with open(actual_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if re.fullmatch(r"[0-9a-fA-F]+", s):
                actual.append(hex_to_signed_int32(s))
            else:
                invalid_actual += 1

    if invalid_expected > 0:
        print(f"⚠️ Warning: skipped {invalid_expected} invalid expected lines.")
    if invalid_actual > 0:
        print(f"⚠️ Warning: skipped {invalid_actual} invalid actual lines.")

    matches = 0
    total = len(expected)
    
    # We only check up to the len of actual received
    check_len = min(len(expected), len(actual))
    
    for i in range(check_len):
        if expected[i] == actual[i]:
            matches += 1
        else:
            # Limit output to first few mismatches and last few
            if matches < 10 or i > total - 5: 
                print(f"  [MISMATCH] Index {i}: Expected {expected[i]}, Got {actual[i]}")

    if len(actual) != len(expected):
        print(f"⚠️ Warning: Result length mismatch. Expected {len(expected)}, Got {len(actual)}")

    if matches == total and total > 0:
        print(f"✅ PASS: All {matches} outputs match perfectly!")
        return True
    else:
        print(f"❌ FAIL: {matches}/{total} matches.")
        return False

def verify_m1(output_dir="./output"):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", action="store_true", help="Force generate new test vectors")
    parser.add_argument("--gen-only", action="store_true", help="Generate vectors only; skip result checking")
    parser.add_argument("--check", action="store_true", help="Only check existing simulation results")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero status on verification failure")
    parser.add_argument("--dim", type=int, default=896, help="Hidden dimension")
    parser.add_argument("--output", type=str, default=output_dir, help="Output directory containing model_ir.json")
    args, _ = parser.parse_known_args()
    output_dir = args.output

    print(f"--- [Stage 4: Verify] DIM={args.dim} ---")
    
    # Step 1: Check files
    ir_path = os.path.join(output_dir, "model_ir.json")
    if not os.path.exists(ir_path):
        print("❌ model_ir.json not found. Run convert.py first.")
        return

    testvec_dir = os.path.join(output_dir, "testvectors")
    input_hex = os.path.join(testvec_dir, "input.hex")

    if args.check:
        ok = check_results(output_dir, dim=args.dim)
        if args.strict and not ok:
            raise SystemExit(1)
        return

    # Step 2: Generate test vectors if forced or missing
    if args.gen or not os.path.exists(input_hex):
        generate_test_vectors(output_dir, dim=args.dim)
    else:
        print("[Verify] Using existing test vectors.")

    if args.gen_only:
        print("[Verify] --gen-only enabled, skipping comparison.")
        return
    
    # Step 3: Check for simulation output
    ok = check_results(output_dir, dim=args.dim)
    if args.strict and not ok:
        raise SystemExit(1)

if __name__ == "__main__":
    verify_m1()
