#!/usr/bin/env bash
# Build llama.cpp for SM89 (L4) with the MMVQ crossover patch.
#
# The patch adds one runtime knob, GGML_MMVQ_MAX, which moves the boundary between
# llama.cpp's two quantized matmul kernels:
#
#   mul_mat_vec_q (MMVQ) - CUDA cores, chosen for ne11 <= MMVQ_MAX_BATCH_SIZE (8)
#   mul_mat_q     (MMQ)  - int8 tensor cores, chosen above that
#
# Speculative verification runs at ne11 = n_draft+1, i.e. 3-6, which lands inside MMVQ.
# On an L4 that is the wrong kernel: measured marginal cost of one extra verified token
# is 12.8 ms under MMVQ and 1.9 ms under MMQ. GGML_MMVQ_MAX=2 pushes verification onto
# MMQ and leaves plain batch-1 decode on MMVQ, where MMVQ is still correct.
#
# Upstream cannot be configured this way: GGML_CUDA_FORCE_MMQ is compile-time only and
# is evaluated inside ggml_cuda_should_use_mmq(), which ggml_cuda_mul_mat() never reaches
# because ggml_cuda_should_use_mmvq() returns true first.
#
# Usage:  bash scripts/build-llamacpp.sh [output-dir]
# Env:    SHA (llama.cpp commit), JOBS
set -euo pipefail

SHA="${SHA:-4df29be4f4c3673f428170fda944a5b19f743bb8}"   # b10454
OUT="${1:-$HOME/lcpp}"
JOBS="${JOBS:-$(nproc)}"
SRC="${SRC:-$HOME/llama.cpp-build}"
PATCH="$(cd "$(dirname "$0")/.." && pwd)/patches/0001-mmvq-runtime-crossover.patch"

command -v nvcc >/dev/null || export PATH=/usr/local/cuda-12.9/bin:$PATH
command -v nvcc >/dev/null || { echo "nvcc not found; install the CUDA toolkit"; exit 1; }

sudo apt-get update -qq
sudo apt-get install -y -qq cmake build-essential ccache git

if [ ! -d "$SRC/.git" ]; then
  git init "$SRC"
  git -C "$SRC" remote add origin https://github.com/ggml-org/llama.cpp.git
fi
git -C "$SRC" fetch --depth 1 origin "$SHA"
git -C "$SRC" checkout -f FETCH_HEAD
git -C "$SRC" apply "$PATCH"

cmake -S "$SRC" -B "$SRC/build" \
  -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CCACHE=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$SRC/build" -j"$JOBS" --target llama-server llama-quantize

mkdir -p "$OUT"
cp -r "$SRC/build/bin" "$OUT/"
echo "built -> $OUT/bin/llama-server"
echo "built -> $OUT/bin/llama-quantize"
