#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIM="${1:-128}"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
STRICT_ATTN="${FORNAX_STRICT_ATTN:-0}"
TORCH_REF="${FORNAX_TORCH_REF:-0}"
NUM_LAYERS="${FORNAX_NUM_LAYERS:-1}"
ENABLE_EMBED="${FORNAX_ENABLE_EMBED:-0}"
ENABLE_FINAL_NORM="${FORNAX_ENABLE_FINAL_NORM:-0}"
ENABLE_LM_HEAD="${FORNAX_ENABLE_LM_HEAD:-0}"
TOKEN_ID="${FORNAX_TOKEN_ID:-0}"
VOCAB_LIMIT="${FORNAX_VOCAB_LIMIT:-}"
DEBUG_SIM="${FORNAX_DEBUG:-0}"
OUT_DIR="${FORNAX_OUTPUT_DIR:-${ROOT_DIR}/output}"
MODEL_DATA_DIR="${FORNAX_MODEL_DATA_DIR:-${ROOT_DIR}/output}"

to_abs_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    echo "$p"
  else
    echo "${ROOT_DIR}/${p#./}"
  fi
}

OUT_DIR="$(to_abs_path "${OUT_DIR}")"
MODEL_DATA_DIR="$(to_abs_path "${MODEL_DATA_DIR}")"
cd "${ROOT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERR] Python venv not found at ${PYTHON_BIN}"
  echo "Run: python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt"
  exit 1
fi

if ! command -v iverilog >/dev/null 2>&1; then
  echo "[ERR] iverilog is not installed or not in PATH."
  exit 1
fi

echo "[STEP] Generate expected vectors (DIM=${DIM})"
echo "[STEP] Rebuild IR/RTL (DIM=${DIM})"
mkdir -p "${OUT_DIR}"
if [[ "${OUT_DIR}" != "${MODEL_DATA_DIR}" ]]; then
  if [[ ! -f "${MODEL_DATA_DIR}/graph.json" ]]; then
    echo "[ERR] graph.json not found in MODEL_DATA_DIR=${MODEL_DATA_DIR}"
    exit 1
  fi
  if [[ ! -d "${MODEL_DATA_DIR}/weights" ]]; then
    echo "[ERR] weights/ not found in MODEL_DATA_DIR=${MODEL_DATA_DIR}"
    exit 1
  fi
  cp "${MODEL_DATA_DIR}/graph.json" "${OUT_DIR}/graph.json"
  rm -rf "${OUT_DIR}/weights"
  cp -R "${MODEL_DATA_DIR}/weights" "${OUT_DIR}/weights"
fi

"${PYTHON_BIN}" -c "from src.converter import ModelConverter; from src.generator import VerilogGenerator; c=ModelConverter('${OUT_DIR}', target_dim=${DIM}, strict_attention=bool(int('${STRICT_ATTN}')), num_layers=int('${NUM_LAYERS}'), enable_embedding=bool(int('${ENABLE_EMBED}')), enable_final_norm=bool(int('${ENABLE_FINAL_NORM}')), enable_lm_head=bool(int('${ENABLE_LM_HEAD}')), token_id=int('${TOKEN_ID}'), vocab_limit=(int('${VOCAB_LIMIT}') if '${VOCAB_LIMIT}' else None)); c.convert(); c.save('${OUT_DIR}'); g=VerilogGenerator('${OUT_DIR}', template_dir='${ROOT_DIR}/templates'); g.generate()"

read -r IN_LEN OUT_LEN <<< "$("${PYTHON_BIN}" -c "import json; ir=json.load(open('${OUT_DIR}/model_ir.json')); ops=ir['ops']; first=ops[0]; last=ops[-1]; in_len=1 if first.get('type')=='embedding_lookup' else int(first.get('dim', first.get('in_features', ${DIM}))); out_len=int(last.get('out_features', last.get('dim', last.get('in_features', ${DIM})))); print(in_len, out_len)")"
echo "[INFO] Testbench lengths: IN_LEN=${IN_LEN}, OUT_LEN=${OUT_LEN}"

mkdir -p "${OUT_DIR}/testbench"
sed -e "s/__DIM__/${DIM}/g" \
    -e "s/__LAYERS__/${NUM_LAYERS}/g" \
    -e "s/__IN_LEN__/${IN_LEN}/g" \
    -e "s/__OUT_LEN__/${OUT_LEN}/g" \
    "${ROOT_DIR}/verify/tb_top.v" > "${OUT_DIR}/testbench/tb_top.v"

rm -f "${OUT_DIR}/testvectors/actual.hex"
"${PYTHON_BIN}" "${ROOT_DIR}/verify/compare.py" --gen --gen-only --dim "${DIM}" --output "${OUT_DIR}"

echo "[STEP] Run RTL simulation"
pushd "${OUT_DIR}/testbench" >/dev/null
if [[ "${DEBUG_SIM}" == "1" ]]; then
  iverilog -DFORNAX_DEBUG -o sim_check ../rtl/*.v tb_top.v
else
  iverilog -o sim_check ../rtl/*.v tb_top.v
fi
vvp sim_check > sim_check.log
popd >/dev/null

echo "[STEP] Compare expected vs actual"
"${PYTHON_BIN}" "${ROOT_DIR}/verify/compare.py" --check --dim "${DIM}" --output "${OUT_DIR}"

if [[ "${TORCH_REF}" == "1" ]]; then
  echo "[STEP] Compare against PyTorch reference"
  "${PYTHON_BIN}" "${ROOT_DIR}/verify/torch_ref_compare.py" --output "${OUT_DIR}" --strict-only
fi

echo "[INFO] Simulation log: ${OUT_DIR}/testbench/sim_check.log"
