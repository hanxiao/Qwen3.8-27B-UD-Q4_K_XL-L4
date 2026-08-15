# Qwen3.8-27B · UD-Q4_K_XL · NVIDIA L4

Serving [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at **~24 tok/s decode with a 65,536-token context on a single NVIDIA L4 (24 GB)**, using the [Unsloth Dynamic v3.0 `UD-Q4_K_XL` GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) and llama.cpp with MTP speculative decoding.

One command from nothing to a running OpenAI-compatible endpoint:

```bash
gcloud config set project <your-project>
bash scripts/provision-ondemand.sh
```

Everything below was measured on the machine this repo provisions, on 2026-08-14, the day Qwen3.8 shipped.

## Why this model is interesting on an L4

Qwen3.8-27B is **not a conventional dense 27B**, and that is the whole story. From `config.json`:

```
num_hidden_layers      64
full_attention_interval 4      -> 16 full-attention layers, 48 linear-attention layers
mtp_num_hidden_layers   1      -> ships with a native MTP draft head
max_position_embeddings 262144
```

Only one layer in four maintains a KV cache. The practical effect on a 24 GB card is large. For reference, benchmarking a genuinely dense 27B (Qwen3.5-27B, Q6_K) on the same hardware in March 2026 produced 10 tok/s and a hard ceiling of 71K context. Qwen3.8-27B does **2.4x the throughput at comparable context**, and the context ceiling is a VRAM budget rather than an architectural wall.

It is still a long way from a sparse MoE of similar size. [Qwen3.6-35B-A3B with MTP on the same L4](https://github.com/hanxiao/Qwen3.6-35B-A3B-MTP-L4) runs 92-100 tok/s, because it activates ~3B parameters per token while this model reads far more of its weights. Pick accordingly: 35B-A3B for raw speed, 27B for a compact dense-ish model with native vision.

## Measured performance

`scripts/bench.sh`, run on the instance. Decode-only (`timings.predicted_per_second`), `cache_prompt: false`, `temperature: 0`, 256 max tokens, 2 runs averaged per workload.

| Workload | Decode tok/s | MTP acceptance |
|---|---|---|
| prose | 26.24 | 0.687 |
| code | 22.72 | 0.580 |
| json | 24.03 | 0.676 |
| chat | 22.90 | 0.662 |
| math | 24.20 | 0.775 |
| multi-turn | 23.54 | 0.729 |
| summarization | 24.25 | 0.757 |
| **MIN** | **22.72** (code) | |

Prefill sits at 95-114 tok/s on short prompts. MTP mean accepted draft length ranges 2.15-2.80 tokens.

Acceptance is workload-dependent and noticeably below what the same MTP setup achieves on 35B-A3B (0.77-0.90). Code generation is the weakest case at 0.58, which is exactly why `code` is the slowest row: rejected drafts are wasted forward passes.

If you measure materially below ~23 tok/s, check that ECC is off and that all layers are resident on the GPU.

## VRAM budget

Measured with the server loaded and idle at `--ctx-size 65536`:

```
21506 MiB used / 24570 MiB total   (ECC disabled)
```

Weights are 17.9 GB. The remaining ~3.6 GB covers the f16 KV cache for 16 full-attention layers, the MTP draft context, and compute buffers. There is headroom left; 65536 was chosen as a comfortable default, not as the ceiling.

**ECC must be off.** `nvidia-smi -e 0` plus a reboot returns roughly 1 GB of VRAM. The provisioning script does this on first boot and reboots once. With ECC enabled this configuration does not fit.

## Available quantizations

From [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). `UD-` prefixed files are Unsloth Dynamic v3.0.

| File | Size | Fits 24 GB L4 |
|---|---|---|
| `UD-IQ2_XXS` | 9.01 GB | yes, large context headroom |
| `UD-Q2_K_XL` | 10.68 GB | yes |
| `UD-Q3_K_XL` | 13.44 GB | yes |
| `Q4_K_M` | 17.11 GB | yes |
| **`UD-Q4_K_XL`** | **17.92 GB** | **yes — this repo's default** |
| `UD-Q5_K_XL` | 20.22 GB | tight, expect a much smaller context |
| `Q6_K` | 22.88 GB | no meaningful room for KV |
| `UD-Q6_K_XL` | 25.92 GB | no |
| `Q8_0` | 29.05 GB | no |

Only `UD-Q4_K_XL` at `--ctx-size 65536` is measured here. The rest of the column is a size-against-capacity judgement, not a benchmark. To try another one:

```bash
HF_FILE=Qwen3.8-27B-UD-Q5_K_XL.gguf CTX=32768 bash scripts/provision-ondemand.sh
```

## Serving configuration

```
--ctx-size 65536 --parallel 1 --flash-attn on -ngl 99
-ub 64 -b 512 --no-mmap --threads 8
--spec-type draft-mtp --spec-draft-n-max 2
--jinja --tools all --metrics
```

`--parallel 1` is deliberate. llama.cpp divides `--ctx-size` across slots, so `--parallel 2` would give two slots 32768 each rather than one slot 65536. A single slot maximises single-stream throughput.

`--spec-draft-n-max 2` matches the depth that proved optimal for this model family elsewhere: depth 1 leaves speedup unclaimed, depth 3 drops acceptance far enough that it becomes a net loss.

Thinking is **on by default**. Chain-of-thought arrives in `reasoning_content`, the answer in `content`. With a small `max_tokens` the entire budget is consumed by reasoning and `content` comes back as an empty string — that is truncation, not a broken server. Use `max_tokens >= 512`, or disable per request:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

## Calling it

```bash
curl -s http://<IP>:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-27B-UD-Q4KXL-MTP",
  "messages": [{"role": "user", "content": "What is 17*23?"}],
  "max_tokens": 600, "temperature": 0
}'
```

Also on the box: a web UI at `http://<IP>:8080`, Prometheus metrics at `/metrics`, and `/props` for the live context and slot configuration.

## Reproducing the benchmark

```bash
ZONE=$(gcloud compute instances list --filter="name=qwen38-27b-l4-od" --format='value(zone)')
gcloud compute scp scripts/bench.sh qwen38-27b-l4-od:~/bench.sh --zone=$ZONE
gcloud compute ssh qwen38-27b-l4-od --zone=$ZONE --command 'sudo apt-get install -y -qq bc; bash ~/bench.sh 127.0.0.1:8080'
```

Run it on the instance. `bench.sh` uses bash 4 associative arrays and will not run under the stock bash 3.2 on macOS.

## Cost

On-demand `g2-standard-8` is roughly $0.81/hr, billed while the instance is RUNNING regardless of load.

```bash
bash scripts/teardown.sh          # stop, keeps disk + model for a fast restart
bash scripts/teardown.sh delete   # full teardown
```

Spot is about $0.24/hr but preemption terminates the instance and releases its external IP mid-session. Use spot only for short throwaway runs.

## Pitfalls

**No `pip` on the deep-learning images.** A startup script that reaches for `huggingface_hub` dies with `pip: command not found` (exit 127) before downloading anything, and the failure is quiet: the instance boots fine, the GPU is healthy, `/opt/models` is simply empty. The fix here is an 8-way ranged `curl` straight off the HuggingFace CDN, which needs nothing beyond curl and pulls 17.9 GB in about 100 seconds — faster than a single-stream `hf_hub_download` regardless. The script verifies the reassembled file against `content-length` before starting the server.

**On-demand L4 capacity is genuinely scarce.** A normal run walks through nine or more zones returning `STOCKOUT` before one lands. Budget ~15 minutes of wall clock to READY. Do not hardcode a zone and do not kill the script while it cycles.

**The external IP changes as zones are retried.** Each retry recreates the instance. Read the IP from `gcloud compute instances list` after the script reports READY; do not trust an IP scraped from earlier output.

**Startup failures are silent from the outside.** `/health` never coming up tells you nothing about why. Go straight to the boot log:

```bash
gcloud compute ssh qwen38-27b-l4-od --zone=$ZONE --command 'sudo tail -50 /var/log/qwen-startup.log'
```

## Files

```
scripts/provision-ondemand.sh   create the L4, fetch the model, start the server, wait for health
scripts/bench.sh                7-workload decode benchmark (run on the instance)
scripts/teardown.sh             stop or delete
```

## Related

- [hanxiao/Qwen3.6-35B-A3B-MTP-L4](https://github.com/hanxiao/Qwen3.6-35B-A3B-MTP-L4) — sparse MoE on the same card, 92-100 tok/s
