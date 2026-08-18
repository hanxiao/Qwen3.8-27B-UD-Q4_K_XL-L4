# Qwen3.8-27B · UD-Q4_K_XL · NVIDIA L4

Qwen3.8-27B serves at 32.9 tok/s decode over a 65,536-token context on a single NVIDIA L4 24 GB, using the [Unsloth Dynamic v3.0 `UD-Q4_K_XL` GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) and llama.cpp with MTP speculative decoding. That is 1.36x the 24.1 tok/s this repository previously served, at identical quantization and with no measured quality regression.

```bash
gcloud config set project <your-project>
bash scripts/provision-ondemand.sh
```

That command creates the instance, fetches the model and the draft head, builds a patched llama.cpp, starts an OpenAI-compatible server and blocks until it answers on `/health`. Every number below was measured on the machine it provisions, on 2026-08-18.

## Summary of the change

Three things moved throughput, in descending order of size. None of them changes the weights, the quantization, or the sampling: speculative decoding emits a drafted token only after the target model verifies it, and the kernel change swaps which CUDA kernel evaluates the same matmul. Both can reorder floating-point reductions and so flip tokens that were near ties; measured accuracy is unchanged, and the Quality section reports the checks.

| Change | Decode (tok/s) | Delta |
| --- | --- | --- |
| Previous configuration (MTP n-max 2, p-min 0.4) | 24.14 | |
| Route speculative verification to the MMQ kernel | 28.05 | +16.2% |
| Draft from a separate, cheaper MTP head | 30.39 | +8.3% |
| Retype that head's output tensor to Q2_K, n-max 5 | 30.7 | +1.1% |
| Sample on the GPU (`-bs`) instead of copying 248,320 logits per verified position to the host | 31.0 | +0.9% |
| Draft the whole chain in one decode, with a 98,304-wide draft sub-head | 32.4 | +4.7% |
| Route Q6_K and IQ4_XS back to MMVQ, per quant type | **32.9** | +1.4% |

The first change is the interesting one, and it is a one-line patch to llama.cpp. The rest of this README explains why, because the reasoning generalizes to any speculative decoding setup on a compute-poor card.

## The cost model

Everything below follows from three measured quantities. Table 1 gives the first.

**Table 1: where the time goes in one decode step.**

| Quantity | Value | How it was measured |
| --- | --- | --- |
| Weights read per target forward pass | 15.75 GiB | GGUF tensor table, minus `token_embd` (682 MiB, a row lookup) and `blk.64` (272 MiB, the MTP block, not in the trunk graph) |
| Target forward pass, batch 1 | 66.9 ms | `--spec-type none` decode at 14.95 tok/s |
| Effective memory bandwidth | 252.8 GB/s | 15.75 GiB / 66.9 ms, against a 300.05 GB/s datasheet peak |

Decode is memory-bound, and llama.cpp reaches 84% of the datasheet peak. It is tempting to call that the practical GDDR6 roofline and stop. It is not, and `tools/bandwidth-probe.cu` shows it: a plain `float4` streaming read over an 8 GiB buffer, 170x the 48 MB L2, sustains **293.4 GB/s, or 97.8% of datasheet**, reproducing at 16 GiB and at both 256 and 512 threads per block.

So the weight-streaming path is leaving about 16% of this card's real bandwidth unused. Two things say that gap is kernel efficiency rather than physics: the streaming probe holds 2040 MHz while drawing only **45.6 W** against a 72 W cap, so streaming alone is neither power- nor clock-limited; and locking the SM clock anywhere between 900 and 2040 MHz moves decode throughput by under 2%, with the card at 63.6 W, so decode is not clock-limited either. The difference is that decode spends power and time on dequantization and on hundreds of dependent kernel boundaries that a single streaming kernel does not have.

Closing that gap is worth more than anything else on the board: at 293.4 GB/s the weight read falls from 66.9 ms to 57.6 ms, which alone would take this configuration to about 36.7 tok/s. Nothing in llama.cpp exposes it as a setting, and it is not something a flag reaches, but it is the largest measured headroom that remains and it is not speculative.

Throughput therefore depends entirely on how many tokens each of those 66.9 ms passes emits:

```
tok/s = mean accepted length / (verify pass cost + drafting cost)
```

## Why verification width was the bottleneck

llama.cpp picks between two quantized matmul kernels by batch width. `ggml_cuda_should_use_mmvq` returns true for `ne11 <= MMVQ_MAX_BATCH_SIZE`, which is 8, and `ggml_cuda_mul_mat` consults it before `ggml_cuda_should_use_mmq`. So every speculative verification batch, which is `n_draft + 1` and therefore 3 to 9 tokens wide, lands on `mul_mat_vec_q`. That kernel runs on CUDA cores. `mul_mat_q` runs on the int8 tensor cores and is never reached.

Table 2 is the consequence. Both columns verify identical batches of identical weights.

**Table 2: cost of one target forward pass against verification width.**

| Verify width | `mul_mat_vec_q` (stock) | `mul_mat_q` (`GGML_MMVQ_MAX=2`) |
| --- | --- | --- |
| 1 | 66.9 ms | n/a, MMVQ is correct at batch 1 |
| 2 | | 74.1 ms |
| 3 | 85.5 ms | 80.6 ms |
| 4 | 111.7 ms | 83.2 ms |
| 8 | | 89.7 ms |
| 12 | | 100.1 ms |
| 16 | | 106.2 ms |

Under MMVQ each additional verified token costs 9 ms going from width 1 to 3 and 26 ms going from 3 to 4. Under MMQ the whole range from 2 to 16 costs about 2 ms per token. That is the difference between a draft depth of 2 being optimal and a draft depth of 5 being optimal, and it is worth 16%.

Fitting both kernels separates two distinct costs:

```
T_MMVQ(B) = 66.9 + ~9.3·(B-1) ms        streams weights at 252.8 GB/s, but each
                                        extra column costs almost a whole pass
T_MMQ(B)  = 66.9 + 9.7 + 1.98·(B-1) ms  a flat 9.7 ms worse at the same width,
                                        then nearly free per extra token
```

MMQ is behind at width 1 and 2 and ahead from width 3, which is exactly where `GGML_MMVQ_MAX=2` puts the boundary. The 9.7 ms is the price of admission: MMQ stages weights through shared memory in tiles rather than streaming them, so it does not reach MMVQ's bandwidth on the same bytes. Since MMQ's smallest tile is 8 columns (`mmq.cuh`, `for (int J = 8; J <= 128; J += 8)`), widths 2 through 8 issue identical tensor-core and unpack work, so that 9.7 ms is genuinely fixed rather than something the draft depth can amortize away.

This is now the largest single removable cost in the cycle, and nothing in llama.cpp removes it. Closing it needs a mixed-precision GEMM that keeps streaming efficiency at M=6, which is what Marlin and Machete do for W4A16 in other stacks and what llama.cpp does not have for K-quants.

The fee is not uniform across quant types, which is worth a little. `GGML_MMVQ_MAX` in the patch takes per-type overrides, and sweeping each of this file's four types independently at draft depth 5 gives:

**Table 2b: routing one quant type back to MMVQ, everything else on MMQ.**

| Routed to MMVQ | Share of weights | Decode (tok/s) |
| --- | --- | --- |
| nothing, all MMQ | | 31.04 |
| Q4_K | 4% | 31.05 |
| **IQ4_XS** | **17%** | **31.76** |
| **Q6_K** | **6%** | **31.83** |
| Q5_K | 72% | 20.51 |

Q5_K must stay on MMQ: it is 72% of the weights and MMVQ's per-column cost lands on all of it, costing a third of throughput. Q6_K and IQ4_XS are small enough that MMVQ's cheaper entry beats its worse slope. Shipping both gives 32.88 against 32.41, measured over the full benchmark.

A third surface is already spent: llama.cpp fuses the Gated DeltaNet snapshot copy into the GDN kernel itself (`ggml_cuda_try_gdn_cache_fusion`), so the separate `ggml_cpy` into `ssm_states_all` that looks like an obvious 3-4 ms of waste in the graph does not actually execute. The kernel still writes `n_draft+1` state snapshots per verification, about 864 MiB at draft depth 5, and removing *that* needs the recurrence factorized rather than snapshotted, which is a much larger change.

Two other tuning surfaces turned out not to be surfaces at all. The MMQ tile table (`mmq-config-ampere.cuh`, which Volta, Turing, Ampere and Ada all share) is not free-form: dropping `I` from 128 to 64 for the J=8 rows builds and then dies with an illegal memory access, because the tile geometry is tied to the MMA fragment layout, and raising `occupancy` from 1 to 2 is inert at 31.14 against 31.21. And `ik_llama.cpp`, the obvious fork to try for better quantized matmuls, has no `draft-mtp` and no fused Gated DeltaNet op, so it cannot run this configuration at all regardless of how fast its kernels are.

This is not reachable by configuration. `GGML_CUDA_FORCE_MMQ` is a compile-time flag evaluated inside `ggml_cuda_should_use_mmq`, which the dispatcher never calls once `ggml_cuda_should_use_mmvq` has returned true. `patches/0001-mmvq-runtime-crossover.patch` adds a `GGML_MMVQ_MAX` environment variable to that one predicate. Setting it to 2 keeps MMVQ for plain batch-1 decode, where MMVQ is genuinely faster, and routes everything wider to MMQ.

## Why the draft head was replaced

The second cost in the model is drafting. Qwen3.8-27B ships a native MTP block, and llama.cpp drafts by running that block plus an output projection once per drafted token. The block is 272 MiB. The output projection is the target's own `output.weight`, which at Q6_K over a 248,320-token vocabulary is 994.6 MiB, so 78% of each draft step is the LM head.

`ggml-org/Qwen3.8-27B-GGUF` publishes the MTP block as a standalone sidecar carrying its own embedding and output tensors at Q4_0. Pointing `--spec-draft-model` at it replaces the Q6_K head with a Q4_0 one. Retyping that file's output tensor to Q2_K with `llama-quantize --output-tensor-type q2_K` shrinks it further. Draft precision is not output precision: the target verifies every token, so a coarser draft head can only change how often a draft is accepted, never what is emitted.

**Table 3: cost of one draft step.**

| Draft head | Output tensor | Read per step | Predicted at 252.8 GB/s | Measured | Mean accepted length, n-max 5 |
| --- | --- | --- | --- | --- | --- |
| Embedded in the target GGUF | 994.6 MiB Q6_K | 1.237 GiB | 5.25 ms | 4.9-5.3 ms | 2.88 |
| `mtp-Qwen3.8-27B-Q4_0.gguf` sidecar | 682.0 MiB Q4_0 | 0.888 GiB | 3.77 ms | 3.94 ms | 2.99 |
| **Same, output tensor retyped Q2_K** | **397.9 MiB Q2_K** | **0.611 GiB** | **2.60 ms** | **2.73 ms** | **3.01** |

Read per step is the output tensor plus the 227.6 MiB MTP block; the embedding is a row lookup and is not read. All three land within 5% of what bandwidth alone predicts, so drafting is bandwidth-bound too and shrinking the head is the only lever on it.

Accepted length does not fall across a 2.0x change in head size, so the whole saving is real. Q3_K sits between Q4_0 and Q2_K in size but measured worse than both, 28.89 tok/s against 29.20 and 29.91 on the same two workloads: its first-position acceptance fell to 0.746, against 0.779 for Q4_0 and 0.763 for Q2_K, which cost more than the bandwidth it saved. Q2_K is the floor that still holds acceptance.

## Why the draft chain runs in one decode

llama.cpp drafts autoregressively: one `llama_decode` per drafted token, each followed by a host round trip to pick the token from a full 248,320-wide logit row. At draft depth 5 that is five GPU calls and five 993 KB transfers per decode cycle.

[PR #27173](https://github.com/ggml-org/llama.cpp/pull/27173) rebuilds that as a single decode that produces the whole chain and picks each token on the GPU, emitting two floats per step instead of a logit row. It also runs the draft against a leading slice of the output tensor rather than the whole thing, on the observation that BPE ids correlate with frequency, so a draft rarely picks a high id. The target still verifies against the full vocabulary, so this narrows what gets drafted and never what gets committed.

The slice width matters, and its default does not suit this model. `LLAMA_SPEC_CHAIN_SUB` defaults to 32768, which is 21% of the 151k vocabulary the PR was tuned against but only 13% of this model's 248,320. At that width mean accepted length falls from 2.89 to 2.64 and the whole gain is given back. Table 4b is the sweep.

**Table 4b: draft sub-head width, at draft depth 5.**

| `LLAMA_SPEC_CHAIN_SUB` | Share of vocab | Cumulative draft time | Mean accepted length | Decode (tok/s) |
| --- | --- | --- | --- | --- |
| off, one decode per token | 100% | 3827 ms | 2.89 | 29.47 |
| 32768 (the default) | 13% | 2093 ms | 2.64 | 29.19 |
| 65536 | 26% | 2281 ms | 2.79 | 30.47 |
| 81920 | 33% | 4663 ms | 2.87 | 31.03 |
| **98304** | **40%** | **4882 ms** | **2.91** | **31.22** |
| 114688 | 46% | 5136 ms | 2.91 | 31.03 |
| 0, full head in one decode | 100% | 3723 ms | 2.90 | 29.69 |

98,304 is the point where accepted length has fully recovered and the head is still only 40% of full width.

Measuring the id distribution of the model's own output explains why, and closes off the obvious next idea. Over 1,792 generated tokens, the fraction falling below an id threshold is 73.6% at 8,192, 88.2% at 32,768, **97.1% at 98,304** and 97.5% at 131,072. The curve has its knee exactly where the sweep put the optimum: below 98,304 acceptance erodes fast, above it there is almost nothing left to buy.

The idea this suggests is FR-Spec: rank the vocabulary by frequency rather than by id, and cover the same 97% in far fewer rows. Measured, it is worse. Ranking tokens by frequency in a neutral corpus (589,184 tokens of public prose and source, 22,734 distinct) and scoring that ranking on held-out generations covers 78.4% at 32,768 rows against id-ordering's 88.2%, and plateaus at 78.4% no matter how many rows are added, because the corpus simply never contains the tail the model uses. Qwen's BPE ids are already ordered by merge frequency, so id-ordering is a frequency ranking built from the tokenizer's own training set, which is far larger than any calibration corpus. Slicing the first N rows is not a crude approximation of FR-Spec here; it is the better version of it. Above it accepted length stops improving and the extra bytes cost throughput; below it acceptance erodes faster than the bandwidth saves. Measured on prose, code and json; the tok/s column is a three-workload subset and so is not directly comparable to Table 4.

Depth still peaks at 5 even with drafting made cheaper: 5, 6, 7 and 8 give 31.2, 29.9, 29.4 and 27.5.

## Serving configuration

```
--ctx-size 65536 --parallel 1 --flash-attn on -ngl 99
-ub 512 -b 512 --no-mmap --threads 8
--spec-type draft-mtp
--spec-draft-model /opt/models/mtp-o2k.gguf --spec-draft-ngl 99
--spec-draft-n-max 5 --spec-draft-n-min 5 --spec-draft-p-min 0.0
-bs --jinja --tools all --metrics
```

with `GGML_MMVQ_MAX=2 GGML_MMVQ_MAX_Q6K=8 GGML_MMVQ_MAX_IQ4XS=8 LLAMA_SPEC_CHAIN=1 LLAMA_SPEC_CHAIN_SUB=98304 LLAMA_SCHED_POOL=8` in the environment.

`--spec-draft-p-min 0.0` reverses the previous configuration, which used 0.4. The confidence gate exists to suppress drafts that are likely to be rejected, and it was worth 4% when a draft step cost 5.3 ms and a rejected draft made verification 26 ms more expensive. At 2.73 ms per draft step and 2.3 ms per unit of verification width, a rejected draft is cheap enough that suppressing it costs more than it saves. Measured at n-max 3 with the cheap head: p-min 0.0 gives 28.71 tok/s, 0.3 gives 27.11, 0.5 gives 26.17, 0.6 gives 25.13.

`--spec-draft-n-min` equal to `--spec-draft-n-max` fixes the draft length, which keeps the verification batch a constant shape and improves CUDA graph reuse.

`-bs` moves sampling onto the GPU. Without it the server copies a full logit row, 248,320 floats or 993 KB, to the host for every verified position; at width 6 that is roughly 6 MB per decode cycle over PCIe gen3 plus a host-side argmax. It is worth about 1%, small because the copy partly overlaps, but it is free.

ECC must be disabled, as before. `nvidia-smi -e 0` plus a reboot returns about 1 GB of VRAM, and this configuration does not fit without it. The provisioning script performs this on first boot.

## Architecture

Qwen3.8-27B uses hybrid attention, and that is why a 27B model is practical on a 24 GB card. Its `config.json` reports 64 layers with `full_attention_interval` of 4, so 16 layers maintain a KV cache and the remaining 48 use linear attention (Gated DeltaNet). The model ships a native MTP draft head, `mtp_num_hidden_layers` of 1, and supports positions up to 262,144.

Because only one layer in four holds a KV cache, context costs roughly a quarter of what it costs on a conventional dense model of this size. A genuinely dense Qwen3.5-27B at Q6_K on this same card reaches 10 tok/s and stops at 71K context.

A sparse MoE of similar size remains faster. [Qwen3.6-35B-A3B with MTP on the same L4](https://github.com/hanxiao/Qwen3.6-35B-A3B-MTP-L4) reaches 92-100 tok/s because it activates roughly 3B parameters per token, so it reads about a fifth as many bytes per token. Choose 35B-A3B for throughput and this model for a compact dense-class model with native vision.

Thinking is enabled by default. Reasoning arrives in `reasoning_content` and the answer in `content`. A small `max_tokens` budget is consumed entirely by reasoning and returns an empty `content`, so use at least 512, or disable thinking per request with `{"chat_template_kwargs": {"enable_thinking": false}}`.

```bash
curl -s http://<IP>:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-27B-UD-Q4KXL-MTP",
  "messages": [{"role": "user", "content": "What is 17*23?"}],
  "max_tokens": 600, "temperature": 0
}'
```

The instance also serves a web UI on port 8080, Prometheus metrics at `/metrics`, and live slot configuration at `/props`.

## Performance

Table 4 reports decode throughput from `scripts/bench.sh`. The metric is `timings.predicted_per_second`, which excludes prompt processing, measured with `cache_prompt` disabled, greedy sampling, 256 max tokens, averaged over two runs per workload. Mean accepted length is tokens emitted per target forward pass, which is the quantity the cost model is written in.

**Table 4: decode throughput and mean accepted length by workload.**

| Workload | Previous (tok/s) | Tuned (tok/s) | Speedup | Tuned mean accepted length |
| --- | --- | --- | --- | --- |
| math | 24.54 | **37.75** | 1.54x | 3.60 |
| summarization | 24.63 | **34.77** | 1.41x | 3.13 |
| multi-turn | 23.69 | **33.38** | 1.41x | 3.01 |
| code | 23.67 | **32.57** | 1.38x | 2.91 |
| json | 22.92 | **31.64** | 1.38x | 2.84 |
| prose | 22.96 | **31.16** | 1.36x | 2.77 |
| chat | 23.24 | **29.45** | 1.27x | 2.65 |
| **average** | **24.18** | **32.96** | **1.36x** | **3.04** |

The previous column is this repository's earlier measurement; re-measuring that configuration with the current harness gave 24.14, so the two are directly comparable. Repeat passes of the shipped configuration gave 32.88 and 32.96; the global-crossover configuration it replaced gave 32.42, 32.41 and 32.49.

Structured reasoning speculates best and open-ended chat worst, which is the usual shape: the gain tracks how predictable the next token is, and `math` reaches 3.53 accepted tokens per pass against `chat` at 2.70.

Throughput is flat against context. At 32,768 / 65,536 / 81,920 the same configuration measures 29.63 / 29.51 / 29.60 tok/s on a two-workload subset, and the full benchmark gives 30.97 at 65,536 against 30.98 at 81,920. Context is a memory question, not a speed one, which is why the shipped default is the one that leaves headroom rather than the one that fits.

## Can this reach 100 tok/s

No, and the arithmetic is worth writing down because it bounds the whole problem rather than this particular implementation.

100 tok/s is 10 ms per token. One target forward pass reads 15.75 GiB of weights, and that read is unavoidable: the model is dense, every byte is touched once per pass, and there is no reuse to exploit. At the 252.8 GB/s llama.cpp currently achieves that pass costs 66.9 ms, so a pass must emit 6.7 tokens. Give it the full 293.4 GB/s the card is measured to deliver and the pass costs 57.6 ms, so it must still emit **5.8 tokens** — and that is before charging anything for drafting or for the extra width of the verification batch.

Mean accepted length is therefore the entire problem, and Table 5 says where it actually is.

**Table 5: mean accepted length, measured and published.**

| Drafter | Mean accepted length | Source |
| --- | --- | --- |
| Native MTP head, draft depth 2 | 2.47 | measured here |
| Native MTP head, draft depth 5 | 3.01 | measured here |
| Native MTP head, draft depth 11 or more | 3.21 | measured here, saturated |
| `DimInfer/Qwen3.8-27B-Dspark-v1`, block 15 | 2.67 | measured here |
| Same, on this quantization | 3.23 | published by its author |
| Same, on Q4_K_M, GSM8K | 4.09 | published by its author, the highest figure found for this model |

The MTP head saturates at 3.21 because its per-position acceptance decays: 0.78, 0.55, 0.35, 0.25, 0.18, 0.10, 0.07. Drafting deeper adds cost and almost no accepted tokens. Nothing published for Qwen3.8-27B reaches 6.7. The gap is a factor of 1.6 against the best figure reported for this model on any quantization, and 2.1 against the best reported on this one.

Three idealized bounds make the headroom concrete. Give drafting away for free and keep the measured verification cost: 3.21 / 100.1 ms is 32.1 tok/s. Give away drafting *and* all verification width, so every pass costs the bare 66.9 ms weight read: 3.21 / 66.9 ms is 48.0 tok/s. Give away the 16% of bandwidth the kernels do not use as well, so the pass costs 57.6 ms: 3.21 / 57.6 ms is 55.7 tok/s. The configuration here reaches 32.9, which is past the first bound. The remaining distance to 100 tok/s is not implementation slack. It is the drafter.

The same arithmetic answers a nearer question, 40 tok/s, which is worth writing down because it is close enough to argue about. At the shipped accepted length of 3.01, 40 tok/s means a 75.2 ms cycle. The current cycle is 92.4 ms: 66.9 ms of weight read, about 9 ms of MMQ admission cost, 7.9 ms of verification width, and 8.3 ms of drafting. Removing the MMQ penalty alone gives 36.2. Removing it and halving drafting gives 38.3. Reaching 40 requires essentially all of the kernel overhead to go, or accepted length to rise from 3.01 to about 3.7. Both are real projects: the first is a Marlin-class kernel for K-quants at M=6, the second is a trained draft head. Neither is a flag.

What would close it is a draft head with mean accepted length near 7, which means a trained head rather than a better-tuned one. The published recipes that reach 5.3 to 5.5 on comparable targets regenerate 12,000 to 40,000 answers from the served quantized target, capture hidden states from the GGUF itself, warm-start from an existing head of the same geometry, and train for 6 to 10 hours on a 96 GB card. That is the identified next step, and it is a training project, not a serving one. A second, cheaper lever is tree drafting: verification width is now nearly free up to 16 tokens, so spending that width on several candidate branches instead of one chain should raise accepted length by roughly a third. llama.cpp has no tree support in any speculative implementation, and adding it needs per-token attention masks that the current API does not expose.

## What did not work

Measured and rejected, so they need not be tried again.

| Attempt | Result |
| --- | --- |
| `DimInfer/Qwen3.8-27B-Dspark-v1`, block-15 DSpark head | 23.96 tok/s over seven workloads, a tie with MTP. Its accepted length is better but it runs two draft-model forwards per cycle, one of which routes target hidden states through host memory, and that erases the gain on a card this compute-poor. |
| `RadixArk` / `erlidev` block-7 DSpark heads | 18.6 to 23.1 tok/s, worse. Per-position acceptance 0.81, 0.39, 0.18. |
| n-gram and suffix drafting (`ngram-mod`, `ngram-simple`, `ngram-map-k`) | Never fires. Over a 256-token generation from a short prompt it produced 0 to 7 draft tokens total. There is no repeated history to mine at this generation length. |
| Locking SM clocks (900, 1200, 1500, 2040 MHz) | Within 2%. Decode is bandwidth-bound, and at 900 MHz the card draws 63.6 W against a 72 W cap, so it is not power-bound either. |
| `-ub` 64 to 512, `-b`, `--threads` 4 to 8, `--flash-attn off`, `--no-op-offload` | All within 0.7% at batch 1. Re-swept on the tuned configuration, `-ub` 16 / 32 / 64 / 128 / 256 gave 29.92 / 28.70 / 29.87 / 29.15 / 29.16, a 1.2 tok/s spread with no trend. `-ub 512` is kept because it helps prefill, which decode does not care about. |
| `GGML_CUDA_GRAPH_OPT=1` | 29.33 against 29.36. Its fusion pattern requires a three-way fan-out at batch 1, which only 16 of this model's 64 blocks have. |
| `GGML_CUDA_DISABLE_FUSION=1` (control) | 28.45 against 29.36, so the existing fusions are worth about 3% and should be left on. |
| llama.cpp master (commit `01818e4`) against the pinned b10454 | 23.38 against 23.60, no difference. |
| Draft head output tensor at Q3_K | 28.89 against Q4_0's 29.20 and Q2_K's 29.91. Acceptance fell faster than bandwidth. |
| Raising `--spec-draft-p-min` with the cheap head | Monotonically worse: 28.71 at 0.0 down to 25.13 at 0.6. |
| PR [#26705](https://github.com/ggml-org/llama.cpp/pull/26705), branchless Q4_K/Q5_K scale unpack | Nothing, and it cannot help here. It removes a per-column re-execution of the scale unpack inside `mul_mat_vec_q`, but this configuration routes verification to `mul_mat_q`. Built and measured: with the patch, MMVQ verification still runs at 23.4 tok/s at depth 3 and 21.9 at depth 5, against 29.6 for MMQ. |
| `DimInfer/Qwen3.8-27B-Dspark-v1`, requantized to Q4_K, at its author's recommended depth | 28.88 / 28.92 / 28.62 at n-max 4 / 5 / 6 with `p-min 0`. Its published mean accepted length of 3.807 at n-max 6 does not reproduce on this workload mix: measured 2.65 / 2.74 / 2.77, against 3.01 for the native MTP head. Their published figures are dominated by math and gsm8k, which speculate far better than chat or json. |

| `xkm/qwen3.8-27b-mtp-head-retrained`, a retrained nextn head | Worse at equal cost. Swapped into the sidecar and requantized to match, it gives mean accepted length 2.85 against the stock head's 2.89. Its published gain is real but requires an F16 head, and on a card this bandwidth-starved the extra 682 MiB per draft step costs more than the acceptance buys. The same artifact wins on Apple Silicon, where the author measured it. |

| An importance matrix for the draft head | Worth ~1% of accepted length and nothing measurable in throughput. Neither `mtp-Qwen3.8-27B-Q4_0.gguf` nor my Q2_K requantization of it carries imatrix metadata, so the draft head was the one uncalibrated component in the pipeline, and Q2_K is the most imatrix-sensitive quantization there is. Computing one (120 chunks of neutral public text, final PPL 3.86) and requantizing lifts accepted length 2.95 to 2.98 and first-position acceptance 0.760 to 0.772, but the file grows 0.4% and the two cancel: 31.83 against 31.84. |
| The retrained nextn head on a matched quantization grid | Still worse. The obvious objection to the earlier result was that MLX affine g64 asymmetric requantized to GGUF Q4_0 g32 symmetric is the worst possible grid mismatch. Redone at Q4_K with the imatrix above, which is the near match: 31.27 and accepted length 2.94, against the stock head's 31.84 and 2.98. Three grids now (Q4_0, Q4_K with imatrix, F16) all land the same way. |
| n-gram drafting at short match lengths | Still never pays. Retested with `ngram-simple` at match length 4 and 6 and `ngram-mod` at 8, on top of the chain configuration: 30.98, 31.43 and 31.58 against a 31.63 control. Short matches fire often enough to preempt MTP (n-gram has fixed dispatch priority) without drafting anything better. |
| `GGML_CUDA_DISABLE_GRAPHS=1`, `--cache-ram 0`, `--ctx-checkpoints 0` | Controls, all negative, which is the useful result: disabling CUDA graphs costs 2.3% (31.01 against 31.71), so graph capture is already working, and neither the prompt cache nor context checkpointing costs anything measurable. |

A note on one misleading measurement, because it cost time. Measuring forward-pass cost by timing prompt processing shows a 3.3x step between width 4 and width 5, which does not exist in the decode path: the same widths measured through actual speculative decoding are flat. Prompt processing and speculative verification take different paths through the server. Measure the path you intend to optimize.

## Memory and context capacity

The draft head is a second model resident in VRAM, so it is paid for in context. Table 6 reports VRAM with the server loaded and idle, on a 24,089 MiB device.

**Table 6: VRAM against context size, tuned configuration.**

| `--ctx-size` | VRAM used | Free |
| --- | --- | --- |
| 8,192 | 18,960 MiB | 5,129 MiB |
| 32,768 | 20,616 MiB | 3,473 MiB |
| **65,536 (default here)** | **22,848 MiB** | **1,722 MiB** |
| 81,920 | 23,946 MiB | 143 MiB, loads and benchmarks but is not safe, see below |
| 90,112 | fails to load | |

**81,920 loads, serves, and benchmarks at full speed, and still crashes.** llama.cpp instantiates a CUDA graph per distinct batch shape, and instantiation allocates. With 143 MiB free, the first request whose shape has not been seen before aborts the process:

```
ggml-cuda.cu:106: CUDA error
CUDA error: out of memory
  cudaGraphInstantiate(&graph->instance, graph->graph, __null, __null, 0)
```

The seven-workload benchmark never triggers it because it replays the same seven shapes, so the number looks fine right up until real traffic arrives. It surfaced only when the 40-prompt accuracy check ran, with `enable_thinking` off and 40 distinct short prompts, and took the server down twice. The fix is headroom, not a smaller graph: 65,536 leaves 1.7 GiB, survives the same stress with zero restarts, and costs nothing measurable in throughput. Treat "fits at idle" as necessary and not sufficient, and validate a context ceiling with varied prompt shapes rather than with the benchmark.

The trade is therefore 65,536 tokens at 32.9 tok/s against 104,192 tokens at 24.1. Sixteen of the 64 layers keep a KV cache, at 64 KiB per token, and the draft context adds 4 KiB per token, so context costs 68 KiB per token against 64 KiB before.

If the full 104,192 matters more than the last 9%, drop `--spec-draft-model` and keep the kernel patch. That configuration measures 28.05 tok/s, needs no second model, and fits the original context: it is `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 3 --spec-draft-p-min 0.0` with `GGML_MMVQ_MAX=2`.

## Quality

`scripts/verify-quality.sh` runs two independent checks against a `--spec-type none` reference: byte-level determinism and accuracy on 40 arithmetic problems with known ground truth. It captures three configurations so the kernel change and speculation can be told apart.

**Table 7: quality checks.**

| Configuration | Accuracy | Byte-identical to reference, per task |
| --- | --- | --- |
| No speculation, stock kernel selection (reference) | 40/40 | reference; identical to itself across two runs |
| No speculation, shipped kernel routing | 40/40 | 4 of 5 |
| Shipped configuration | 40/40 | 1 of 5 |

Accuracy is unchanged. Byte-level output is not, and neither divergence is a regression.

Speculative decoding is lossless by construction, since a drafted token is emitted only after the target verifies it, but accepting a drafted token changes the batch shape of the forward pass and therefore the order of floating-point reduction, which flips tokens that were near ties. The middle row isolates the same effect for the kernel change alone, with speculation held out. Note that at batch 1 both configurations use `mul_mat_vec_q` regardless of the crossover, so the one prompt that diverges there does so through *prompt processing*: the prompt is wide enough to cross the boundary, the two kernels accumulate the same products in a different order, and the resulting KV cache differs in the last bits and carries into generation. Running any one configuration twice is byte-identical, so the server itself is deterministic and the divergence is a property of the reduction order, not of run-to-run noise.

If you need bit-exact reproducibility against a previously captured reference, run `--spec-type none` without `GGML_MMVQ_MAX` at roughly 15 tok/s.

## Reproducing

```bash
ZONE=$(gcloud compute instances list --filter="name=qwen38-27b-l4-od" --format='value(zone)')
gcloud compute scp scripts/bench.sh scripts/verify-quality.sh qwen38-27b-l4-od:~/ --zone=$ZONE
gcloud compute ssh qwen38-27b-l4-od --zone=$ZONE --command 'sudo apt-get install -y -qq bc; bash ~/bench.sh 127.0.0.1:8080'
gcloud compute ssh qwen38-27b-l4-od --zone=$ZONE --command 'BIN=/opt/llama.cpp/build bash ~/verify-quality.sh'
```

`verify-quality.sh` stops the running server, cycles three configurations and leaves the last one up, so restart the unit afterwards with `sudo systemctl restart qwen-server`.

Run the benchmark on the instance. It uses bash 4 associative arrays, which the bash 3.2 shipped with macOS does not support.

To rebuild llama.cpp yourself, on any CUDA machine:

```bash
bash scripts/build-llamacpp.sh ~/lcpp
```

The build pins llama.cpp at b10454, applies `patches/0001-mmvq-runtime-crossover.patch`, and targets SM89 only. On 32 vCPUs it takes about four minutes; on the 8 vCPUs of a `g2-standard-8` allow twenty. Building on a separate CPU instance and copying `bin/` to `/opt/llama.cpp/build/bin` is faster and keeps the benchmark machine idle.

The patch is one predicate in `ggml/src/ggml-cuda/mmvq.cu` plus a log-level change in `common/speculative.cpp` that promotes llama.cpp's own speculative statistics (mean accepted length, per-position acceptance, cumulative drafting time) from trace to info. Every number in the cost model above came from that line; without it the only way to see drafting cost is `-lv 6`, which logs every draft candidate and slows generation by 30%.
## Network access

The server binds `0.0.0.0` and has no authentication, so the firewall is the only thing standing between the model and the internet. The provisioning script defaults to `INTERNAL=1`, which tags the instance `llama-internal` and admits `tcp:8080` from RFC1918 only. `INTERNAL=0` opens it to `0.0.0.0/0` through the `llama-server` tag, which is convenient for a throwaway benchmark and wrong for anything that stays up.

If an instance was created with `INTERNAL=0` and you want to close it later, move the tag rather than adding a rule:

```bash
gcloud compute firewall-rules create allow-llama-internal-8080 \
  --network=default --direction=INGRESS --action=ALLOW --rules=tcp:8080 \
  --target-tags=llama-internal --source-ranges=10.128.0.0/9

gcloud compute instances add-tags    qwen38-27b-l4-od --zone=$ZONE --tags=llama-internal
gcloud compute instances remove-tags qwen38-27b-l4-od --zone=$ZONE --tags=llama-server
```

Removing `llama-server` is the part that matters. Firewall rules are additive, so keeping both tags leaves the permissive rule in force and the narrow rule changes nothing.

The external IP stays attached. It carries outbound traffic only, which the startup script needs to reach Hugging Face, GitHub and the Ubuntu archives; a VM with no external IP and no Cloud NAT will boot into an empty `/opt/models` and no compiler. Verify from both sides, because a closed port and a dead server look identical from outside:

```bash
curl -m 10 http://$EXTERNAL_IP:8080/health                 # expect a timeout
gcloud compute ssh <another-vm> --command \
  'curl -s http://<internal-ip>:8080/health'               # expect {"status":"ok"}
```

Administrative access runs over IAP (`gcloud compute ssh --tunnel-through-iap`), so no inbound port needs to be open for operations.

## Quantizations

Table 8 lists the files available in [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). Files prefixed `UD-` use Unsloth Dynamic v3.0. Only `UD-Q4_K_XL` is measured here; the fit column compares file size against device capacity.

**Table 8: quantizations and fit on a 24 GB L4.**

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

The provisioning script takes the target file through instance metadata:

```bash
gcloud compute instances add-metadata qwen38-27b-l4-od --zone=$ZONE \
  --metadata=qwen-hf-file=Qwen3.8-27B-UD-Q5_K_XL.gguf,qwen-ctx=49152
```

The draft head is tied to the model family, not to the target's quantization, so it does not change with the target file.


## Cost

An on-demand `g2-standard-8` costs approximately $0.81/hr and bills while the instance is RUNNING, independent of load. At 32.9 tok/s that is 146,000 tokens per dollar, against 107,000 before.

```bash
bash scripts/teardown.sh          # stop, preserving disk and models
bash scripts/teardown.sh delete   # full teardown
```

Spot instances cost about $0.24/hr, but preemption terminates the instance and releases its external IP. Reserve spot for short runs.

## Operational notes

The deep-learning images do not provide `pip` on PATH. A startup script that downloads through `huggingface_hub` exits 127 before transferring anything, and the failure is quiet: the instance boots, the GPU is healthy and `/opt/models` is empty. This repository fetches both GGUFs with ranged `curl`, which requires only curl and verifies each reassembled file against `content-length` before the server starts.

The instance builds llama.cpp on first boot, which is the long pole at roughly 20 minutes on 8 vCPUs. It runs concurrently with the two downloads. If you provision often, build once with `scripts/build-llamacpp.sh` on a larger CPU instance and copy `bin/` to `/opt/llama.cpp/build/bin`.

The server runs under systemd as `qwen-server`, not under docker, because the patched binary is built locally. `journalctl -u qwen-server` has the serving log; `/var/log/qwen-startup.log` has the provisioning log.

On-demand L4 capacity is scarce. A typical run walks through nine or more zones returning `STOCKOUT` before one succeeds, so let the script cycle. Each retry recreates the instance, so read the external IP from `gcloud compute instances list` once the script reports READY.

A stopped instance restarts with `bash scripts/start.sh`, keeping its disk, the models and the build, but the zone does not hold capacity for it. Restarts return `STOCKOUT` exactly as creation does, and a stopped instance cannot change zones, so the script retries in place until an L4 frees up. Serving flags come from instance metadata (`qwen-ctx`, `qwen-nmax`, `qwen-mmvq-max`), so a restart picks up whatever the metadata currently says.

When `/health` never comes up, the boot log carries the reason:

```bash
gcloud compute ssh qwen38-27b-l4-od --zone=$ZONE --command 'sudo tail -80 /var/log/qwen-startup.log'
```

## Files

| Path | Purpose |
| --- | --- |
| `scripts/provision-ondemand.sh` | create the L4, then hand off to `startup.sh`, and wait for health |
| `scripts/startup.sh` | on-instance: ECC off, build llama.cpp, fetch both models, install the systemd unit |
| `scripts/build-llamacpp.sh` | build a patched, SM89-only llama.cpp anywhere with a CUDA toolkit |
| `patches/0001-mmvq-runtime-crossover.patch` | adds `GGML_MMVQ_MAX`, the kernel crossover knob |
| `scripts/bench.sh` | seven-workload decode benchmark |
| `scripts/sweep-spec.sh` | speculative decoding parameter sweep; still drives the previous docker path, so point it at `/opt/llama.cpp/build/bin/llama-server` before use |
| `scripts/max-context.sh` | binary search for the largest usable context; same caveat |
| `scripts/verify-quality.sh` | determinism and accuracy checks |
| `scripts/start.sh` | restart a stopped instance, retrying through STOCKOUT |
| `scripts/teardown.sh` | stop or delete |

## Related

[hanxiao/Qwen3.6-35B-A3B-MTP-L4](https://github.com/hanxiao/Qwen3.6-35B-A3B-MTP-L4) serves a sparse MoE on the same card at 92-100 tok/s.

## License

The scripts in this repository are Apache-2.0. The model weights are covered by their own licence: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and the [Unsloth GGUF conversion](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) are both Apache-2.0. The draft head comes from [ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF), same licence.
