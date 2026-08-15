# Qwen3.8-27B · UD-Q4_K_XL · NVIDIA L4

Qwen3.8-27B serves at 23 tok/s decode with a 65,536-token context on a single NVIDIA L4 24 GB, using the [Unsloth Dynamic v3.0 `UD-Q4_K_XL` GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) and llama.cpp with MTP speculative decoding. The card holds up to 102,400 tokens of context at this quantization.

```bash
gcloud config set project <your-project>
bash scripts/provision-ondemand.sh
```

That command creates the instance, fetches the model, starts an OpenAI-compatible server and blocks until it answers on `/health`. Every number below was measured on the machine it provisions, on 2026-08-14.

## Architecture

Qwen3.8-27B is a hybrid-attention model, which is what makes it viable on a 24 GB card. Its `config.json` reports 64 layers with `full_attention_interval` of 4, so 16 layers maintain a KV cache and the remaining 48 use linear attention. The model also ships a native MTP draft head, `mtp_num_hidden_layers` of 1, and supports positions up to 262,144.

One layer in four holding a KV cache is what removes the context ceiling that a conventional dense model of this size runs into. A genuinely dense Qwen3.5-27B at Q6_K on this same card reaches 10 tok/s and stops at 71K context. Qwen3.8-27B sustains 2.4 times that throughput at comparable context, and its context limit is set by the VRAM budget rather than by the architecture.

A sparse MoE of similar size remains faster. [Qwen3.6-35B-A3B with MTP on the same L4](https://github.com/hanxiao/Qwen3.6-35B-A3B-MTP-L4) reaches 92-100 tok/s because it activates roughly 3B parameters per token. Choose 35B-A3B for throughput and this model for a compact dense-class model with native vision.

## Serving configuration

```
--ctx-size 65536 --parallel 1 --flash-attn on -ngl 99
-ub 64 -b 512 --no-mmap --threads 8
--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.4
--jinja --tools all --metrics
```

llama.cpp divides `--ctx-size` across slots, so `--parallel 1` gives a single slot the full context instead of splitting it. The two speculative parameters are measured optima, reported in the tuning section below.

The default of 65,536 leaves 3 GB of VRAM free. Raise it with `CTX=102400 bash scripts/provision-ondemand.sh` to use the full capacity measured below.

ECC must be disabled. Turning it off with `nvidia-smi -e 0` and rebooting returns about 1 GB of VRAM, and this configuration does not fit without it. The provisioning script performs this on first boot.

Thinking is enabled by default. Reasoning arrives in `reasoning_content` and the answer in `content`. A small `max_tokens` budget is consumed entirely by reasoning and returns an empty `content`, so use at least 512, or disable thinking per request with `{"chat_template_kwargs": {"enable_thinking": false}}`.

```bash
curl -s http://<IP>:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-27B-UD-Q4KXL-MTP",
  "messages": [{"role": "user", "content": "What is 17*23?"}],
  "max_tokens": 600, "temperature": 0
}'
```

The instance also serves a web UI on port 8080, Prometheus metrics at `/metrics`, and live slot configuration at `/props`.

## Memory and context capacity

Table 1 reports VRAM measured with the server loaded and idle. Weights account for 17.9 GB at every setting; the difference is the f16 KV cache for the 16 full-attention layers, the MTP draft context and compute buffers.

**Table 1: VRAM against context size, on a 24,570 MiB device.**

| `--ctx-size` | VRAM used | Free |
| --- | --- | --- |
| 65,536 (default) | 21,500 MiB | 3,070 MiB |
| 90,112 | 23,138 MiB | 1,432 MiB |
| **102,400 (maximum)** | **23,956 MiB** | **614 MiB** |
| 105,472 | fails to serve | |

`scripts/max-context.sh` locates this limit by binary search. A context counts as working only when the server both reaches `/health` and completes a generation, because llama.cpp allocates the KV cache at load time while some compute buffers are sized on the first decode. A context that loads can still fail under traffic.

The limit was confirmed under load rather than at idle. An 89,125-token prompt at `--ctx-size 102400` returned a correct answer, held decode at 25.4 tok/s with prefill at 293 tok/s, and peaked at 23,962 MiB.

## Performance

Table 2 reports decode throughput from `scripts/bench.sh`, run on the instance under the shipped configuration. The metric is `timings.predicted_per_second`, which excludes prompt processing, measured with `cache_prompt` disabled, greedy sampling, 256 max tokens, averaged over two runs per workload.

**Table 2: Decode throughput and MTP acceptance by workload.**

| Workload | Decode (tok/s) | MTP acceptance |
| --- | --- | --- |
| summarization | **24.63** | 0.810 |
| math | 24.54 | 0.803 |
| multi-turn | 23.69 | 0.798 |
| code | 23.67 | 0.810 |
| chat | 23.24 | 0.762 |
| prose | 22.96 | 0.736 |
| json | 22.92 | 0.725 |

Prefill reaches 95-114 tok/s on short prompts.

The L4 provides 300 GB/s of memory bandwidth, so reading 17.9 GB of weights once per token bounds plain autoregressive decode at approximately 16.8 tok/s. Throughput above that bound comes from speculative decoding. A measurement materially below 23 tok/s indicates that ECC is still enabled or that some layers are not resident on the GPU.

## Tuning

`scripts/sweep-spec.sh` reproduces this section. The sweep covers draft-side and scheduling parameters, which change how tokens are proposed but not which tokens are ultimately emitted. Target-side KV quantization is excluded because it trades quality for memory, and the objective here is speed at fixed quality.

Table 3 reports the one parameter that improved throughput. `--spec-draft-p-min` suppresses drafting when the draft head is not confident, which avoids issuing a forward pass whose result would be rejected.

**Table 3: Effect of the draft confidence threshold. Average is over five workloads.**

| p-min | Average (tok/s) | Code (tok/s) | Code acceptance |
| --- | --- | --- | --- |
| off | 22.58 | 20.66 | 0.576 |
| 0.30 | 22.92 | 22.48 | 0.686 |
| 0.35 | 23.47 | 23.90 | 0.793 |
| **0.40** | **23.54** | **23.90** | **0.810** |
| 0.45 | 22.90 | 20.42 | 0.670 |
| 0.50 | 23.23 | 23.68 | 0.817 |
| 0.60 | 22.30 | 22.28 | 0.888 |
| 0.80 | 21.03 | 21.60 | 0.912 |

The gain concentrates on the workload that was previously slowest: code rises from 20.66 to 23.90 tok/s, and the spread across workloads narrows from 22.7-26.2 to 22.9-24.6.

Acceptance is a diagnostic and not an objective. A threshold of 0.80 produces the highest acceptance in Table 3 and nearly the lowest throughput, because it suppresses productive drafts along with unproductive ones.

Table 4 records the remaining parameters at their measured settings, so they need not be swept again.

**Table 4: Parameters that leave throughput unchanged or reduce it.**

| Parameter | Setting | Average (tok/s) | Note |
| --- | --- | --- | --- |
| `--spec-draft-n-max` | **2** | **21.62** | optimum |
| | 3 | 20.70 | acceptance decays faster than draft length grows |
| | 4 | 19.15 | |
| | 6 | 16.11 | |
| | 1 | 21.82 | one token per verification caps the gain |
| `--spec-type` | `draft-mtp` | 21.93 | optimum |
| | `draft-mtp,ngram-cache` | 20.92 | MTP already covers these patterns |
| | `draft-mtp,ngram-map-k` | 21.79 | |
| | `ngram-cache` | 14.87 | acceptance 0.000 on prose |
| `-ub` | 64 to 512 | 21.64-21.67 | decode issues one token per pass |
| `--spec-draft-n-min` | 1, 2 | 23.04, 22.28 | no gain over the default |
| `--no-spec-draft-backend-sampling` | | 23.10 | within noise |
| `--no-op-offload` | | 22.96 | within noise |

Re-testing draft depths 3 and 4 with p-min active did not recover them, at 23.56 and 22.95 against 23.54 for depth 2.

Two upstream leads do not apply here. `GGML_CUDA_MMVQ_MAX` from llama.cpp [PR#26079](https://github.com/ggml-org/llama.cpp/pull/26079) tunes the mvq to MMQ crossover and reports 25-49% gains on K-quant dense decode, but its own measurements show mvq ahead at batch sizes of 5 and below, and MTP verification here runs at batch 3. The PR is also unmerged, and the symbol is absent from the `server-cuda` image. The MTP performance regression in [issue #23774](https://github.com/ggml-org/llama.cpp/issues/23774) is specific to the Vulkan backend.

## Quality

`scripts/verify-quality.sh` runs two independent checks. Speculative decoding is lossless by construction, since a drafted token is emitted only after the target model verifies it, and the checks confirm this holds in practice.

Byte-level output is not identical to `--spec-type none`. Two controls locate the cause. Running one configuration twice is byte-identical on all five tasks, so the server is deterministic. Plain MTP without p-min already diverges from no-speculation, so the flag is not responsible. Accepting a drafted token changes the batch shape of the forward pass and therefore the order of floating-point reduction, which flips tokens that were near ties. This property is intrinsic to speculative decoding.

Correctness is the operative test. Table 5 reports accuracy on 40 arithmetic problems with known ground truth.

**Table 5: Verifiable accuracy across configurations.**

| Configuration | Accuracy |
| --- | --- |
| No speculation | 40/40 |
| MTP n=2 | 40/40 |
| MTP n=2 with p-min 0.4 | 40/40 |

Run `--spec-type none` at roughly 16-17 tok/s when bit-exact reproducibility matters more than throughput.

## Quantizations

Table 6 lists the files available in [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). Files prefixed `UD-` use Unsloth Dynamic v3.0. Only `UD-Q4_K_XL` is measured here; the fit column compares file size against device capacity.

**Table 6: Quantizations and fit on a 24 GB L4.**

| File | Size | Fit |
| --- | --- | --- |
| `UD-IQ2_XXS` | 9.01 GB | large context headroom |
| `UD-Q2_K_XL` | 10.68 GB | fits |
| `UD-Q3_K_XL` | 13.44 GB | fits |
| `Q4_K_M` | 17.11 GB | fits |
| **`UD-Q4_K_XL`** | **17.92 GB** | **default here** |
| `UD-Q5_K_XL` | 20.22 GB | fits, with roughly 2.3 GB less for KV |
| `Q6_K` | 22.88 GB | no room for KV cache |
| `UD-Q6_K_XL` | 25.92 GB | exceeds capacity |
| `Q8_0` | 29.05 GB | exceeds capacity |

```bash
HF_FILE=Qwen3.8-27B-UD-Q5_K_XL.gguf CTX=32768 bash scripts/provision-ondemand.sh
```

## Reproducing the benchmark

```bash
ZONE=$(gcloud compute instances list --filter="name=qwen38-27b-l4-od" --format='value(zone)')
gcloud compute scp scripts/bench.sh qwen38-27b-l4-od:~/bench.sh --zone=$ZONE
gcloud compute ssh qwen38-27b-l4-od --zone=$ZONE --command 'sudo apt-get install -y -qq bc; bash ~/bench.sh 127.0.0.1:8080'
```

Run the benchmark on the instance. It uses bash 4 associative arrays, which the bash 3.2 shipped with macOS does not support.

## Cost

An on-demand `g2-standard-8` costs approximately $0.81/hr and bills while the instance is RUNNING, independent of load.

```bash
bash scripts/teardown.sh          # stop, preserving disk and model
bash scripts/teardown.sh delete   # full teardown
```

Spot instances cost about $0.24/hr, but preemption terminates the instance and releases its external IP. Reserve spot for short runs.

## Operational notes

The deep-learning images do not provide `pip` on PATH. A startup script that downloads through `huggingface_hub` exits 127 before transferring anything, and the failure is quiet: the instance boots, the GPU is healthy and `/opt/models` is empty. This repository fetches the GGUF with an 8-way ranged `curl`, which requires only curl, completes 17.9 GB in about 100 seconds, and verifies the reassembled file against `content-length` before the server starts.

On-demand L4 capacity is scarce. A typical run walks through nine or more zones returning `STOCKOUT` before one succeeds, so allow roughly 15 minutes to reach READY and let the script cycle. Each retry recreates the instance, so read the external IP from `gcloud compute instances list` once the script reports READY.

When `/health` never comes up, the boot log carries the reason:

```bash
gcloud compute ssh qwen38-27b-l4-od --zone=$ZONE --command 'sudo tail -50 /var/log/qwen-startup.log'
```

## Files

| Path | Purpose |
| --- | --- |
| `scripts/provision-ondemand.sh` | create the L4, fetch the model, start the server, wait for health |
| `scripts/bench.sh` | seven-workload decode benchmark |
| `scripts/sweep-spec.sh` | speculative decoding parameter sweep |
| `scripts/max-context.sh` | binary search for the largest usable context |
| `scripts/verify-quality.sh` | determinism and accuracy checks |
| `scripts/teardown.sh` | stop or delete |

## Related

[hanxiao/Qwen3.6-35B-A3B-MTP-L4](https://github.com/hanxiao/Qwen3.6-35B-A3B-MTP-L4) serves a sparse MoE on the same card at 92-100 tok/s.
