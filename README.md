# Qwen3.8-27B · UD-Q4_K_XL · NVIDIA L4

| | |
| --- | --- |
| Model | Qwen3.8-27B, [Unsloth Dynamic v3.0 `UD-Q4_K_XL` GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), the 2026-08-19 rebuild, 17,559,178,144 bytes |
| Weights | `UD-Q4_K_XL`, 16.35 GiB on disk, 15.35 GiB read per forward pass |
| KV cache | `q4_0`, 18 KiB per token across the 16 full-attention layers |
| Context | **226,048 tokens**, planned by `--fit-target 768 -ub 256`, not pinned |
| Decode, prose | **40.0 tok/s** |
| Decode, seven workloads | **37.7 tok/s** at the full window, 38.3 at a 32k window (math 48.7, code 28.5) |
| Prefill | **556 tok/s** at a 22k prompt, 281 at 87k, 188 at 152k |
| Mean accepted length | **3.63** on the benchmark, **5.51** on GSM8K-shaped prompts, rising to **4.57** at 109k of context |
| Drafter | [DFlash 2](https://inco.ai/blog/dflash2/) `Q4_K_M`, 1.14 GiB, block drafting at `n-max 7`, `p-min` 0 |
| Kernel routing | `GGML_MMVQ_MAX=2`, all types on MMQ |
| Build | llama.cpp [PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342) plus `patches/0001` |
| Checkpoints | `--ctx-checkpoints 4 --checkpoint-min-step 16384` |
| Batching | `-ub 256 -b 2048`, `--parallel 1` |
| GPU | NVIDIA L4 24 GB, ECC off, 72 W cap, 23.0 of 24.0 GiB resident |
| Quality | 40/40 on the arithmetic set, unchanged against `--spec-type none` |

Decode is the seven-workload benchmark, four repetitions, alternated against a control; the run-to-run spread is about 0.4 tok/s. Prefill and the accepted length at depth come from growing a single conversation to 152,361 tokens with the prefix reused, which is the shape the deployment serves. Accepted length is reported per target forward pass, so 3.63 means one verification emits 3.63 tokens.

The table above is the shipped configuration. The sections below give the measurement that fixes each value in it, and the cost model those measurements follow from.

```bash
gcloud config set project <your-project>
bash scripts/provision-ondemand.sh
```

That command creates the instance, fetches the model and the draft head, builds a patched llama.cpp, starts an OpenAI-compatible server and blocks until it answers on `/health`. Every number below was measured on the machine it provisions, on 2026-08-17 and 2026-08-18.

The same card will not serve this model at 100 tok/s, and [the limit is arithmetic](#throughput-ceiling). The 15.75 GiB weight read is irreducible at this quantization, so 100 tok/s requires 6.7 accepted tokens per target forward pass, against 3.01 measured here and 4.09 as the highest figure published for this model. Below draft depth 11 that requirement exceeds what the depth can supply at perfect acceptance.

## Summary

Five elements carry the configuration, in descending order of what each is worth. None of them
changes the weights below the `UD-Q4_K_XL` floor, and none changes the sampling: speculative decoding
emits a drafted token only after the target verifies it, and the kernel change swaps which CUDA
kernel evaluates the same matmul. Both reorder floating-point reductions and so flip tokens that were
near ties; measured accuracy is unchanged, and the Quality section reports the checks.

| Element | Decode (tok/s) | Worth |
| --- | --- | --- |
| A stock deployment: the native MTP head at draft depth 2, `p-min` 0.4 | 24.14 | |
| Route speculative verification to the MMQ kernel | 28.05 | +16.2% |
| Draft from a separate, cheaper head with a Q2_K output tensor | 30.7 | +9.4% |
| Sample on the GPU (`-bs`) instead of copying 248,320 logits per verified position to the host | 31.0 | +0.9% |
| Replace the native head with a block-diffusion drafter | 35.99 | +15.0% |
| Take the 2026-08-19 rebuild of the weights | **37.86** | +5.2% |

The first element is a one-line patch to llama.cpp. The rest of this README explains why, because the
reasoning generalizes to any speculative decoding setup on a compute-poor card.

The first four rows were measured on the build this repository used before the block drafter, which
is a different llama.cpp branch. The block-drafter row is therefore quoted from the same-binary
comparison that isolates it: 31.30 tok/s for the native head against 35.99 for the block drafter,
with only the drafter changed.

Two elements this repository previously shipped are no longer in the configuration and are kept in
the text because the measurements behind them are still the reason. Chain drafting, worth 4.7%, is an
optimization for drafting one token per decode and has nothing to do for a drafter that emits a whole
block per pass. Returning `Q6_K` and `IQ4_XS` to MMVQ, worth 1.4%, was tuned at a narrower
verification width and loses 3.4 tok/s at the width a block drafter verifies.

## Cost model

Everything below follows from three measured quantities. Table 1 gives the first.

**Table 1: where the time goes in one decode step.**

| Quantity | Value | How it was measured |
| --- | --- | --- |
| Weights read per target forward pass | 15.75 GiB | GGUF tensor table, minus `token_embd` (682 MiB, a row lookup) and `blk.64` (272 MiB, the MTP block, not in the trunk graph) |
| Target forward pass, batch 1 | 66.9 ms | `--spec-type none` decode at 14.95 tok/s |
| Effective memory bandwidth | 252.8 GB/s | 15.75 GiB / 66.9 ms, against a 300.05 GB/s datasheet peak |

Decode is memory-bound, and llama.cpp reaches 84% of the datasheet peak. It is tempting to call that the practical GDDR6 roofline and stop. It is not, and `tools/bandwidth-probe.cu` shows it: a plain `float4` streaming read over an 8 GiB buffer, 170x the 48 MB L2, sustains **293.4 GB/s, or 97.8% of datasheet**, reproducing at 16 GiB and at both 256 and 512 threads per block.

So the weight-streaming path is leaving about 16% of the real bandwidth of this card unused. Two measurements attribute that gap to kernel efficiency. The streaming probe holds 2040 MHz while drawing **45.6 W** against a 72 W cap, so streaming alone is neither power- nor clock-limited. Locking the SM clock anywhere between 900 and 2040 MHz then moves decode throughput by under 2%, with the card at 63.6 W, so decode is not clock-limited either. The difference is that decode spends power and time on dequantization and on hundreds of dependent kernel boundaries that a single streaming kernel does not have.

Closing that gap would be worth more than anything else on the board: at 293.4 GB/s the weight read falls from 66.9 ms to 57.6 ms, which alone would put this configuration near 36.7 tok/s. So it is worth knowing why it does not close.

The first guess is the MMVQ launch configuration, since sm_89 matches no tuned table and falls through to `MMVQ_PARAMETERS_GENERIC`. Rebuilding with the decode case (`ncols_dst == 1`) retuned four ways measures 14.99 tok/s at the shipped `nwarps=4, rows_per_block=1`, against 14.90 at `nwarps=8`, 14.98 at `rows_per_block=2`, 14.97 at both, and 14.93 at `nwarps=2`. A 0.6% spread. The table is not the bottleneck.

The power numbers explain it, and the clock telemetry pins down the mechanism. The **memory clock never moves**: it sits at 6251 MHz, its only supported value, through both workloads, so the memory subsystem is always at full speed and there is no memory-clock headroom to find. What the power cap throttles is the **SM clock**, which oscillates between 1230 and 1935 MHz during decode while `SW Power Cap` reads Active and draw stays pinned near 72 W. The streaming probe, by contrast, holds 2040 MHz at **45.6 W**.

So the chain is: dequantization burns power, the cap throttles the SM clock to compensate, and SMs at a throttled clock with dequant work interleaved cannot issue memory requests fast enough to keep the bus full. Decode sits at the **72 W cap**. Decode is doing everything the probe does plus unpacking K-quant scales and running dot products, and on a 72 W card that extra ALU work competes with the memory subsystem for the same budget. That also resolves what looked like a contradiction earlier: locking the SM clock down to 900 MHz drops power to 63.6 W and leaves throughput flat, because the clock is not the binding constraint in that range either. Between roughly 900 MHz and the capped 1335 MHz the configuration is pinned at about 253 GB/s from two directions at once, and the 40 GB/s difference from a pure stream is the power cost of dequantization.

That makes the 16% structural on this part, and it is not available. It is recoverable only by doing less ALU work per byte read, which means a different kernel family, not a different setting.

Throughput therefore depends entirely on how many tokens each of those 66.9 ms passes emits:

```
tok/s = mean accepted length / (verify pass cost + drafting cost)
```

## Kernel selection at verification width

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

MMQ is behind at width 1 and 2 and ahead from width 3, which is exactly where `GGML_MMVQ_MAX=2` puts the boundary. The 9.7 ms is a fixed admission cost: MMQ stages weights through shared memory in tiles rather than streaming them, so it does not reach the MMVQ bandwidth on the same bytes. Since the MMQ smallest tile is 8 columns (`mmq.cuh`, `for (int J = 8; J <= 128; J += 8)`), widths 2 through 8 issue identical tensor-core and unpack work, so that 9.7 ms is fixed, and the draft depth cannot amortize it away.

This is the largest single removable cost in the cycle, and nothing in llama.cpp removes it. Closing it needs a mixed-precision GEMM that keeps streaming efficiency at M=6, which is what Marlin and Machete do for W4A16 in other stacks and what llama.cpp does not have for K-quants.

The fee is not uniform across quant types, which is worth a little. `GGML_MMVQ_MAX` in the patch takes per-type overrides, and sweeping each of the four types in this file independently at draft depth 5 gives:

**Table 3: routing one quant type back to MMVQ, everything else on MMQ.**

| Routed to MMVQ | Share of weights | Decode (tok/s) |
| --- | --- | --- |
| nothing, all MMQ | | 31.04 |
| Q4_K | 4% | 31.05 |
| **IQ4_XS** | **17%** | **31.76** |
| **Q6_K** | **6%** | **31.83** |
| Q5_K | 72% | 20.51 |

Q5_K must stay on MMQ: it is 72% of the weights and the MMVQ per-column cost lands on all of it, costing a third of throughput. Q6_K and IQ4_XS are small enough that the MMVQ cheaper entry beats its worse slope. Shipping both gives 32.88 against 32.41, measured over the full benchmark.

A third surface is not available: llama.cpp fuses the Gated DeltaNet snapshot copy into the GDN kernel itself (`ggml_cuda_try_gdn_cache_fusion`), so the separate `ggml_cpy` into `ssm_states_all` that appears in the graph does not execute. The kernel still writes `n_draft+1` state snapshots per verification, about 864 MiB at draft depth 5, and removing *that* needs the recurrence factorized instead of snapshotted, which is a much larger change.

Two adjacent surfaces are fixed by the hardware. The MMQ tile table (`mmq-config-ampere.cuh`, shared by Volta, Turing, Ampere and Ada) is not free-form. Dropping `I` from 128 to 64 for the J=8 rows builds and then dies with an illegal memory access, because the tile geometry is tied to the MMA fragment layout. Raising `occupancy` from 1 to 2 is inert, 31.14 against 31.21. And `ik_llama.cpp`, the obvious fork to try for better quantized matmuls, has no `draft-mtp` and no fused Gated DeltaNet op, so it cannot run this configuration at all regardless of how fast its kernels are.

This is not reachable by configuration. `GGML_CUDA_FORCE_MMQ` is a compile-time flag evaluated inside `ggml_cuda_should_use_mmq`, which the dispatcher never calls once `ggml_cuda_should_use_mmvq` has returned true. `patches/0001-mmvq-runtime-crossover.patch` adds a `GGML_MMVQ_MAX` environment variable to that one predicate. Setting it to 2 keeps MMVQ for plain batch-1 decode, where MMVQ is genuinely faster, and routes everything wider to MMQ.

## The native draft head

This section describes the drafter this configuration used before the block drafter, and the
measurements are kept because they are the reason the block drafter wins: they establish that
drafting is bandwidth-bound and that draft precision is not output precision. The shipped
configuration drafts with DFlash 2 instead, measured at 35.99 tok/s against 31.30 for the head below
on the same binary.

The second cost in the model is drafting. Qwen3.8-27B ships a native MTP block, and llama.cpp drafts by running that block plus an output projection once per drafted token. The block is 272 MiB. The output projection is the `output.weight` of the target, which at Q6_K over a 248,320-token vocabulary is 994.6 MiB, so 78% of each draft step is the LM head.

`ggml-org/Qwen3.8-27B-GGUF` publishes the MTP block as a standalone sidecar carrying its own embedding and output tensors at Q4_0. Pointing `--spec-draft-model` at it replaces the Q6_K head with a Q4_0 one. Retyping the output tensor of that file to Q2_K with `llama-quantize --output-tensor-type q2_K` shrinks it further. Draft precision is not output precision: the target verifies every token, so a coarser draft head can only change how often a draft is accepted, never what is emitted.

**Table 4: cost of one draft step.**

| Draft head | Output tensor | Read per step | Predicted at 252.8 GB/s | Measured | Mean accepted length, n-max 5 |
| --- | --- | --- | --- | --- | --- |
| Embedded in the target GGUF | 994.6 MiB Q6_K | 1.237 GiB | 5.25 ms | 4.9-5.3 ms | 2.88 |
| `mtp-Qwen3.8-27B-Q4_0.gguf` sidecar | 682.0 MiB Q4_0 | 0.888 GiB | 3.77 ms | 3.94 ms | 2.99 |
| **Same, output tensor retyped Q2_K** | **397.9 MiB Q2_K** | **0.611 GiB** | **2.60 ms** | **2.73 ms** | **3.01** |

Read per step is the output tensor plus the 227.6 MiB MTP block; the embedding is a row lookup and is not read. All three land within 5% of what bandwidth alone predicts, so drafting is bandwidth-bound too and shrinking the head is the only lever on it.

The block is the other half of the draft read, 227.6 MiB against 157 MiB for the sub-head, and a cheaper quantization does not help it. Rebuilding it at Q3_K with the imatrix gives 30.67 tok/s and accepted length 2.89, against 31.77 and 2.95 at Q4_0. Cumulative draft time also rises, 5143 ms against 4831, despite the smaller file, because the Q3_K dequantization kernel is slower per byte than the Q4_0 one. Q4_0 block with a Q2_K output tensor is the optimum on both axes.

Accepted length does not fall across a 2.0x change in head size, so the whole saving is real. Q3_K sits between Q4_0 and Q2_K in size but measured worse than both, 28.89 tok/s against 29.20 and 29.91 on the same two workloads: its first-position acceptance fell to 0.746, against 0.779 for Q4_0 and 0.763 for Q2_K, which cost more than the bandwidth it saved. Q2_K is the floor that still holds acceptance.

## Chain drafting

Chain drafting is not in the shipped configuration. It optimizes drafting one token per decode, and
a block drafter emits a whole block in a single pass, so there is nothing left for it to remove. It
lives on a different llama.cpp branch from the one that carries DFlash 2. The measurements are kept
because the sub-head observation below is the same mechanism that makes a cheap draft output tensor
work at all.

llama.cpp drafts autoregressively: one `llama_decode` per drafted token, each followed by a host round trip to pick the token from a full 248,320-wide logit row. At draft depth 5 that is five GPU calls and five 993 KB transfers per decode cycle.

[PR #27173](https://github.com/ggml-org/llama.cpp/pull/27173) rebuilds that as a single decode that produces the whole chain and picks each token on the GPU, emitting two floats per step instead of a logit row. It also runs the draft against a leading slice of the output tensor rather than the whole thing, on the observation that BPE ids correlate with frequency, so a draft rarely picks a high id. The target still verifies against the full vocabulary, so this narrows what gets drafted and never what gets committed.

The slice width matters, and its default does not suit this model. `LLAMA_SPEC_CHAIN_SUB` defaults to 32768, which is 21% of the 151k vocabulary the PR was tuned against but only 13% of the 248,320 of this model. At that width mean accepted length falls from 2.89 to 2.64 and the whole gain is given back. Table 5 is the sweep.

**Table 5: draft sub-head width, at draft depth 5.**

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

Measuring the id distribution of the model output explains why, and closes off the obvious next idea. Over 1,792 generated tokens, the fraction falling below an id threshold is 73.6% at 8,192, 88.2% at 32,768, **97.1% at 98,304** and 97.5% at 131,072. The curve has its knee exactly where the sweep put the optimum: below 98,304 acceptance erodes fast, above it there is almost nothing left to buy.

The idea this suggests is FR-Spec: rank the vocabulary by frequency rather than by id, and cover the same 97% in far fewer rows. Measured, it is worse. Ranking tokens by frequency in a neutral corpus of 589,184 tokens of public prose and source, 22,734 of them distinct, covers 78.4% at 32,768 rows, against 88.2% for id-ordering. It plateaus at 78.4% however many rows are added, because that corpus never contains the tail the model uses. the BPE ids of Qwen are already ordered by merge frequency, so id-ordering is a frequency ranking built from the training set of the tokenizer, which is far larger than any calibration corpus. Slicing the first N rows is not a crude approximation of FR-Spec here; it is the better version of it. Above it accepted length stops improving and the extra bytes cost throughput; below it acceptance erodes faster than the bandwidth saves. Measured on prose, code and json; the tok/s column is a three-workload subset and so is not directly comparable to Table 6.

Depth still peaks at 5 even with drafting made cheaper: 5, 6, 7 and 8 give 31.2, 29.9, 29.4 and 27.5.

## Serving configuration

```
--parallel 1 --flash-attn on --fit on --fit-target 768
--ctx-checkpoints 4 --checkpoint-min-step 16384
-ub 256 -b 2048 --cache-type-k q4_0 --cache-type-v q4_0 --no-mmap --threads 8
--spec-type draft-dflash
--spec-draft-model /opt/models/dflash2-q4km.gguf --spec-draft-ngl 99
--spec-draft-n-max 7 --spec-draft-n-min 1 --spec-draft-p-min 0.0
-bs --jinja --tools all --metrics
```

with `GGML_MMVQ_MAX=2 GGML_MMVQ_MAX_Q6K=2 GGML_MMVQ_MAX_IQ4XS=2 LLAMA_SCHED_POOL=8` in the
environment, against a binary built from llama.cpp [PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342)
with `patches/0001-mmvq-runtime-crossover.patch` applied.

`--spec-draft-n-max 7` is the block size minus one. The drafter declares `dflash.block_size = 8` and
llama.cpp clamps the request to `block_size - 1`, warning when it does. The clamp mutates a copy while
`need_n_rs_seq()` sizes the target recurrent state from the value as given, so asking for 8 or 9
drafts exactly 7 tokens and spends 299 or 598 MiB of VRAM to do it. Ask for 7.

`--spec-draft-n-min 1` matters more than it looks. The check is `result.size() < n_min`, and it
discards the whole draft when it fails, so a value above what the drafter produces silently disables
speculation rather than shortening it.

`--spec-draft-p-min 0.0` is the default and the only correct value here: the DFlash 2 branch returns
before the confidence gate is reached, so the parameter is never read and the block is drafted in
full every round.

`-bs` moves sampling onto the GPU. Without it the server copies a full logit row, 248,320 floats or
993 KB, to the host for every verified position; at width 8 that is roughly 8 MB per decode cycle
over PCIe gen3 plus a host-side argmax.

ECC must be disabled. `nvidia-smi -e 0` plus a reboot returns about 1 GB of VRAM, and this
configuration does not fit without it. The provisioning script performs this on first boot.

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

Table 6 reports decode throughput from `scripts/bench.sh`. The metric is `timings.predicted_per_second`, which excludes prompt processing, measured with `cache_prompt` disabled, greedy sampling, 256 max tokens, averaged over two runs per workload. Mean accepted length is tokens emitted per target forward pass, which is the quantity the cost model is written in.

**Table 6: decode throughput and mean accepted length by workload.**

| Workload | Stock (tok/s) | Shipped (tok/s) | Ratio | Mean accepted length |
| --- | --- | --- | --- | --- |
| math | 24.54 | **37.75** | 1.54x | 3.60 |
| summarization | 24.63 | **34.77** | 1.41x | 3.13 |
| multi-turn | 23.69 | **33.38** | 1.41x | 3.01 |
| code | 23.67 | **32.57** | 1.38x | 2.91 |
| json | 22.92 | **31.64** | 1.38x | 2.84 |
| prose | 22.96 | **31.16** | 1.36x | 2.77 |
| chat | 23.24 | **29.45** | 1.27x | 2.65 |
| **average** | **24.18** | **32.96** | **1.36x** | **3.04** |

Both columns are measured with the same harness on the same machine. Repeat passes of the shipped configuration fall between 32.4 and 33.0.

Structured reasoning speculates best and open-ended chat worst, which is the usual shape: the gain tracks how predictable the next token is, and `math` reaches 3.53 accepted tokens per pass against `chat` at 2.70.

Generation length changes the answer, and in a way worth stating explicitly because it is the difference between this benchmark and most published figures. Acceptance is worst in the first tokens after a prompt and climbs as the model settles into an answer, so a 256-token cap measures the least favourable part of every generation. Raising only the cap, changing nothing else:

**Table 7: the same configuration at a higher token cap.**

| Workload | 256 max tokens | 1024 max tokens | Acceptance at 1024 |
| --- | --- | --- | --- |
| summarization | 34.22 | **46.51** | 0.672 |
| math | 37.16 | 39.54 | 0.536 |
| prose | 30.58 | 34.28 | 0.438 |
| code | 31.95 | 33.10 | 0.416 |
| json | 31.09 | 31.57 | 0.388 |
| chat | 29.15 | 27.91 | 0.321 |
| multi-turn | 32.87 | 26.30 | 0.290 |
| **average** | **32.43** | **34.17** | |

The average moves 5.4%, but the spread roughly doubles, and two workloads get *worse*: `chat` and `multi-turn` run past their natural answer into open-ended continuation, which speculates badly, while `summarization` stays inside the source text and reaches 0.672 acceptance and 46.51 tok/s. Three workloads (`prose`, `math`, and `summarization` at 2048) stop on their own before the cap, so their rows compare a truncated answer against a complete one, so the two lengths are not of the same text.

The headline number in Table 6 stays the 256-token one. It is the harder measurement and the one this repository has always reported, and moving to a friendlier cap to claim a larger number would make the figure incomparable with its own history. It is worth knowing that sustained generation on this configuration runs nearer 34, and that a summarization-shaped workload runs nearer 46.

Throughput is flat against context. At 32,768 / 65,536 / 81,920 the same configuration measures 29.63 / 29.51 / 29.60 tok/s on a two-workload subset, and the full benchmark gives 30.97 at 65,536 against 30.98 at 81,920. Context is a memory question, not a speed one, which is why the shipped default is the one that leaves headroom, not the one that fits.

That flatness is a statement about *allocated* context, not about occupied context, and the two come apart. This model keeps 16 full-attention layers at 4 KV heads and head dimension 256, so an f16 KV cache costs 4 KiB per token per layer, 64 KiB per token in total, and every resident token is re-read by attention on every forward pass. The benchmark runs short prompts and 256-token generations, so a few hundred tokens are resident and the KV read is about 30 MiB per pass, 0.2% of the 15.75 GiB weight read and comfortably under a millisecond. That is why it does not appear in the cost model. Filled up, it would: 8,192 resident tokens cost 0.5 GiB per pass (3.2% of the weight read, about 2 ms) and 65,536 resident tokens cost 4.0 GiB per pass, 25% on top of the weight read and roughly 17 ms of the cycle. Nothing in this repository is tuned for that regime, and KV cache quantization, which buys nothing measurable here, is the first thing to reach for there.

## Other 24 GB cards

[`Don-Chad/ninfer-3090`](https://github.com/Don-Chad/ninfer-3090) is a purpose-built C++20/CUDA engine for this exact model on a single RTX 3090, and it is the most useful external calibration point available: same model, same 24 GB class, a from-scratch engine instead of llama.cpp. Its single-user result is **71.00 tok/s decode** at 61.13% MTP acceptance, from a 16.96 GiB groupwise artifact.

The RTX 3090 has 936 GB/s of memory bandwidth against 300 for this L4, a factor of 3.12. Scaling this configuration by bandwidth alone gives 32.87 x 3.12 = **102.6 tok/s**, against their measured 71.00. That comparison needs no assumption about their accepted length; it says only that per unit of memory bandwidth, this stack is delivering more than a hand-written engine does.

Normalizing the other way, and taking their MTP3 acceptance to mean an accepted length near 2.83:

| | bandwidth | weight read | cycle | accepted length | tok/s | cycle / weight read |
| --- | --- | --- | --- | --- | --- | --- |
| NInfer, RTX 3090, one user | 936 GB/s | 19.5 ms | 39.9 ms | 2.83 | 71.00 | 2.05x |
| **this, L4** | **300 GB/s** | **59.7 ms** | **92.5 ms** | **3.04** | **32.87** | **1.55x** |

The last column reports how much a stack spends above the unavoidable weight read. This configuration spends 1.55x, theirs 2.05x. Cross-projecting each stack onto the hardware of the other, holding its own overhead and accepted length: their engine on an L4 lands near 23 tok/s, and **this configuration on a 3090 lands near 101 tok/s**.

**100 tok/s for this model at this quantization is reachable, and this software reaches it: doing so needs about 913 GB/s of memory bandwidth.** A 3090 (936), a 4090 (1008) or an L40S (864, ~95 tok/s) are in that class. An L4 at 300 GB/s is not, and no amount of software closes a 3x hardware gap. The work in this repository is portable to those cards unchanged; the patch, the draft head and the flags are all bandwidth-agnostic.

## Comparison with a tree-drafting engine

llama.cpp has no tree drafting, and this file argues above that tree drafting is the most promising unexploited idea here: verification width is nearly free up to the MMQ tile boundary, so spending that width on several candidate branches instead of one chain should raise accepted length. [Lucebox](https://github.com/Luce-Org/lucebox) is an engine built on exactly that premise. It ships DDTree, publishes speedups of 2x to 5.6x over vendored llama.cpp on Qwen 3.6 27B, and takes GGUF input, so it can be pointed at the model file in this repository unchanged. It was worth measuring.

It runs. Lucebox loads `Qwen3.8-27B-UD-Q4_K_XL.gguf`, recognizes `family:qwen35`, and engages DDTree with the Qwen3.6 DFlash drafter in 18.7 GiB. All numbers below use one streaming probe against both servers, counting deltas between first and last token so prefill is excluded, which reproduces the `predicted_per_second` of llama.cpp to within 2%.

The first result is the one that localizes everything else:

| | decode |
| --- | --- |
| llama.cpp, no speculation | 14.98 tok/s |
| **Lucebox, no drafter** | **14.77 tok/s** |

A dead heat, which is what a 15.75 GiB weight read at a fixed bandwidth should produce in any engine. Whatever Lucebox wins elsewhere, it does not win it in the kernels. Their advantage is entirely in the speculative layer, so the question becomes whether their tree beats the chain used here.

**Table 8: DDTree budget sweep, Qwen3.6 DFlash drafter against the Qwen3.8 target, on prose / code / json.**

| DDTree budget | Decode | Tokens per target forward |
| --- | --- | --- |
| 4 | 21.47 | 2.12 |
| 8 | 21.03 | 2.23 |
| 12 | 20.67 | 2.23 |
| 16 | 23.00 | 2.53 |
| 22 (their default) | 24.13 | 2.64 |
| **32** | **25.17** | |
| 48 | 22.17 | |
| 64 | 21.37 | |
| 96 | out of memory | |
| **this repository, same probe** | **31.35** | **3.01** |

The tree works exactly as advertised: tokens committed per target forward climb monotonically with budget, 2.12 to 2.64, and llama.cpp cannot do that at all. It is still not enough. At its best budget Lucebox reaches 25.17 against 31.35 here, 20% behind, because 2.64 tokens per forward loses to 3.01 for the native MTP head and the wider verification costs more on a card with 300 GB/s and a 72 W cap than it returns.

The reason is not the engine, it is the drafter, and the drafter situation is specific to this model being three days old. Lucebox drives DFlash and DSpark drafters; it has no MTP path, so the one strong drafter this model ships cannot be fed to its tree. No Qwen3.8 DFlash drafter exists on its side, only the Qwen3.6 one used above. Training a native one is not obviously the fix either: the only published SpecForge-trained Qwen3.8 DFlash drafter reports 1.81 accepted tokens per step, *worse* than the 2.02 of the Qwen3.6 weights transplanted, so the artifact that exists is undertrained.

Two caveats, because the comparison is not perfectly clean. Lucebox served these requests with `thinking=false` while the configuration here reasons, so the two are generating different text from the same prompts; and Lucebox was given 32,768 context against 65,536 here, which this file measures as worth nothing in throughput but which is not nothing in fairness. Neither moves a 20% gap.

The useful conclusion is a decomposition. Tree drafting is the better verification mechanism and is worth roughly half a token per forward on this evidence. A well-matched drafter is worth more. This model gives away the second to get the first, and on this hardware that trade loses.

## Power limit

The L4 is a single-slot card with no auxiliary power connector, so it lives inside the PCIe slot budget. `nvidia-smi` reports min 40 W, default 72 W, **max 72 W**: the cap cannot be raised, only lowered. That turns out to matter more than it looks, and it is a second wall standing independently of the drafter.

Sweeping the cap while leaving clocks on auto:

**Table 9: decode throughput against the power cap.**

| Power cap | Decode | SM clock |
| --- | --- | --- |
| 45 W | 12.02 tok/s | 315 MHz |
| 55 W | 19.69 tok/s | 495 MHz |
| 65 W | 27.46 tok/s | 870 MHz |
| 72 W (stock) | 30.86 tok/s | 1455 MHz |

Throughput tracks the cap closely, and the first three steps are nearly straight: +7.67, +7.77, then +3.40 over a 7 W step, against 10 W for the others. The curve is bending at the top but has not flattened.

The comparison that gives this meaning is the streaming probe in `tools/bandwidth-probe.cu`, which does nothing but read memory. It reaches 293.4 GB/s at a 50 W cap and gains nothing from the remaining 22 W: 293.4 / 293.5 / 293.6 GB/s at 50 / 60 / 72 W. **Pure memory streaming on this card is done at 50 W. Quantized decode uses every watt of 72.** The difference is dequantization: unpacking Q5_K and IQ4_XS blocks into something the tensor cores can multiply is ALU work, it is on the critical path of every byte read, and it is what consumes the other 22 W.

The same mechanism accounts for the gap between the 293.4 GB/s the card can stream and the 252.8 GB/s llama.cpp actually achieves during decode. It is not kernel sloppiness. The memory clock never moves; the SM clock throttles. Decode is buying its bandwidth with a power budget it is simultaneously spending on unpacking.

The obvious follow-up is whether a better voltage/frequency point exists that the governor is not choosing, since a lower clock runs at a lower voltage and power goes as roughly f·V². Locking the graphics clock at a fixed 72 W cap:

**Table 10: decode throughput against locked SM clock, cap fixed at 72 W.**

| SM clock | Decode | Power drawn |
| --- | --- | --- |
| auto | 31.14 tok/s | 72.05 W |
| 2040 MHz | 31.41 tok/s | 70.96 W |
| 1800 MHz | 31.22 tok/s | 72.50 W |
| 1600 MHz | 31.29 tok/s | 72.35 W |
| 1400 MHz | 31.35 tok/s | 72.21 W |
| 1200 MHz | 31.27 tok/s | 72.10 W |
| 1000 MHz | 29.65 tok/s | 69.07 W |

Every setting from 1200 MHz up returns the same throughput inside a 0.3 tok/s spread. There is no better operating point to find: the governor has already found it, and the only thing an explicit lock achieves is to make things worse once the clock drops far enough to become the binding constraint itself, which happens between 1200 and 1000 MHz.

Sampling the card at 100 ms through 118 seconds of decode says how tight the cap is:

| | |
| --- | --- |
| power drawn | mean 69.75 W, median 71.48, p95 73.01, against a 72 W cap |
| samples within 1 W of the cap | 65.6% |
| SM clock | mean 1286 MHz, p05 1080, p95 1830 |
| memory clock | 6251 MHz, constant, never throttles |
| GPU utilization | 92.1% |

Decode sits at the cap about two thirds of the time, and the SM clock swings across a 750 MHz band absorbing it while the memory clock never moves once. That is the signature of a power limit being paid for out of the compute side of the chip: the memory system is never the thing being slowed down, the unpacking is. It also shows the governor spending part of its time below the 1200 MHz knee, which is the most plausible reason every locked setting in Table 10 edges slightly above `auto`. The effect is a consistent 0.5%, which is inside run-to-run noise, so nothing is shipped on it.

The accurate statement about power on this card is narrower than "power-bound" and more specific than "bandwidth-bound". Decode is pinned at the 72 W cap; the cap is genuinely binding, because lowering it costs throughput immediately; and no software knob redistributes that budget, because the hardware governor is already spending it well. It is a hardware ceiling with no software handle on it, which is why it appears here and no tuning entry follows from it.

It also sharpens the comparison with the RTX 3090 in the section above. That card has 3.12x the memory bandwidth of an L4 and 4.9x the power budget, 350 W against 72 W. For an FP16 model the bandwidth ratio would be the whole story. For a K-quantized model, where every byte read must also be unpacked, the power ratio is part of it too, and the L4 is the more starved of the two on that axis.

## Throughput ceiling

This card will not serve the model at 100 tok/s. The arithmetic below bounds the problem for any implementation on it.

100 tok/s is 10 ms per token. One target forward pass reads 15.75 GiB of weights, and that read is unavoidable: the model is dense, every byte is touched once per pass, and there is no reuse to exploit. At the 252.8 GB/s decode achieves, that pass costs 66.9 ms, so a pass must emit **6.7 tokens**, before charging anything for drafting or for the extra width of the verification batch. The power section above explains why 252.8, rather than the 293.4 GB/s the card can stream, is the right figure to divide by, and why that gap is a hardware tax, not kernel slack.

Mean accepted length therefore determines the result, and Table 11 reports it.

**Table 11: mean accepted length, measured and published.**

| Drafter | Mean accepted length | Source |
| --- | --- | --- |
| Native MTP head, draft depth 2 | 2.47 | measured here |
| Native MTP head, draft depth 5 | 3.01 | measured here |
| Native MTP head, draft depth 11 or more | 3.21 | measured here, saturated |
| `DimInfer/Qwen3.8-27B-Dspark-v1`, block 15 | 2.67 | measured here |
| Same, on this quantization | 3.23 | published by its author |
| Same, on Q4_K_M, GSM8K | 4.09 | published by its author, the highest figure found for this model |

The MTP head saturates at 3.21 because its per-position acceptance decays: 0.78, 0.55, 0.35, 0.25, 0.18, 0.10, 0.07. Drafting deeper adds cost and almost no accepted tokens. Nothing published for Qwen3.8-27B reaches 6.7. The gap is a factor of 1.6 against the best figure reported for this model on any quantization, and 2.1 against the best reported on this one.

One consequence of the power section above deserves stating here, because it moves the ceiling. The 16% gap between the 293.4 GB/s the card streams and the 252.8 GB/s decode achieves is not kernel slack waiting to be recovered: it is the power cost of unpacking, paid on a card that spends two thirds of decode pinned at a 72 W cap it cannot exceed. Bypassing L1 was the cheapest test of the alternative explanation, that the gap is a cache-efficiency artifact, and it came back negative. So the realistic weight-read term is 66.9 ms, not 57.6, and bounds built on 57.6 describe a different GPU, not a better version of this software.

Rebuilding the near bound on that basis: a perfect K-quant kernel would remove the 9 ms MMQ admission fee and shrink verification width, but it cannot remove the weight read or drafting. 66.9 ms of weight read, about 4 ms of irreducible width for six verified tokens, and 8.3 ms of drafting is a 79.2 ms cycle, which at the shipped accepted length of 3.01 is **38.0 tok/s**. That is the target for kernel work alone on this card, and it lands below 40. Passing 40 needs accepted length above roughly 3.2 as well, and passing 100 needs it near 6.6.

The configuration here reaches 32.4 to 33.0 tok/s, which is past the first of those bounds. The remaining distance is not implementation slack; it is the drafter.

The same arithmetic answers a nearer question, 40 tok/s, which is worth writing down because it is close enough to argue about. At the shipped accepted length of 3.01, 40 tok/s means a 75.2 ms cycle. The current cycle is 92.1 ms: 66.9 ms of weight read, about 9 ms of MMQ admission cost, 7.9 ms of verification width, and 8.3 ms of drafting. Removing the MMQ penalty alone gives 36.2. Removing it and halving drafting gives 38.3. Reaching 40 requires essentially all of the kernel overhead to go, or accepted length to rise from 3.01 to about 3.7. Both are real projects: the first is a Marlin-class kernel for K-quants at M=6, the second is a trained draft head. Neither is a flag.

One more form of the question is worth settling, because "not with this drafter" invites "then with which drafter?". Sweep the draft depth and ask what accepted length each depth would need, against the ceiling that depth itself imposes, since accepted length can never exceed depth + 1:

**Table 12: what 100 tok/s would require at each draft depth.**

| Draft depth | Verify width | Cycle | Accepted length needed for 100 tok/s | Ceiling (depth + 1) |
| --- | --- | --- | --- | --- |
| 5 | 6 | 92.1 ms | 9.21 | 6 |
| 7 | 8 | 98.6 ms | 9.86 | 8 |
| 9 | 10 | 114.1 ms | 11.41 | 10 |
| 11 | 12 | 120.5 ms | 12.05 | 12 |
| 15 | 16 | 133.5 ms | 13.35 | 16 |

Up to depth 11 the requirement exceeds what the depth can physically supply even at 100% acceptance, so 100 tok/s is arithmetically impossible there for any drafter whatsoever. From depth 15 it stops being impossible and starts being absurd instead. A 133.5 ms cycle needs 13.35 accepted tokens out of 15 drafted, which at uniform per-position acceptance means **about 97.5% at every one of fifteen consecutive positions** (p = 0.95 yields 11.20, p = 0.97 yields 12.86, p = 0.975 yields 13.32). The head measured here runs 0.78, 0.55, 0.35, 0.25, 0.18, 0.10, 0.07 and is under 10% by position six; the best first-position acceptance published for this model anywhere is 0.855, decaying from there.

A drafter that held 97.5% for fifteen straight tokens would be reproducing the target output almost exactly, at which point it is the model and the 27B target is redundant. 100 tok/s here is not blocked by a missing optimization; it is blocked by requiring a drafter good enough to make the thing it drafts for unnecessary.

The same table read at 40 tok/s is much friendlier, and is the useful number to take away:

| Accepted length | Source | Best depth | Throughput | With a perfect kernel |
| --- | --- | --- | --- | --- |
| 3.01 | measured here | 5 | 32.7 | 36.2 |
| 4.09 | best published for this model, on Q4_K_M / GSM8K | 7 | **41.5** | 45.7 |
| 5.5 | best published EAGLE3-class training recipe | 9 | 48.2 | 52.4 |

So 40 tok/s needs a drafter no better than the best figure already published for this model, and roughly 50 tok/s is what a state-of-the-art trained head would be worth here. Both are drafter projects, and neither requires a single further kernel change.

What would close it is a draft head with mean accepted length near 7, which means a trained head, not a better-tuned one. The published recipes that reach 5.3 to 5.5 on comparable targets regenerate 12,000 to 40,000 answers from the served quantized target, capture hidden states from the GGUF itself, warm-start from an existing head of the same geometry, and train for 6 to 10 hours on a 96 GB card. That is the identified next step, and it is a training project, not a serving one. A second, cheaper lever is tree drafting: verification width is now nearly free up to 16 tokens, so spending that width on several candidate branches instead of one chain should raise accepted length by roughly a third. llama.cpp has no tree support in any speculative implementation, and adding it needs per-token attention masks that the current API does not expose.

## Configurations not used

Measured and rejected, so they need not be tried again.

| Attempt | Result |
| --- | --- |
| `DimInfer/Qwen3.8-27B-Dspark-v1`, block-15 DSpark head | 23.96 tok/s over seven workloads, a tie with MTP. Its accepted length is better but it runs two draft-model forwards per cycle, one of which routes target hidden states through host memory, and that erases the gain on a card this compute-poor. |
| `RadixArk` / `erlidev` block-7 DSpark heads | 18.6 to 23.1 tok/s, worse. Per-position acceptance 0.81, 0.39, 0.18. |
| n-gram and suffix drafting (`ngram-mod`, `ngram-simple`, `ngram-map-k`) | Never fires. Over a 256-token generation from a short prompt it produced 0 to 7 draft tokens total. There is no repeated history to mine at this generation length. |
| Bypassing L1 for global loads (`-Xptxas -dlcm=cg`) | Worse, and the reason matters more than the result. Decode reads 15.75 GiB of weights with no reuse whatsoever, so every L1 tag lookup and fill on that stream is spent for nothing, and on a card pinned at its power cap that waste should be worth reclaiming. Built with the flag and alternated against the deployed binary twice: 30.72 and 30.77 tok/s against 31.38 and 31.12, a consistent 1.6% loss. The flag is indiscriminate, and the loads it also bypasses are the ones that *do* have reuse: the q8_1-quantized activation tile is read by every block in the grid. A surgical version, `__ldcg` on the weight tiles only, would avoid that, but the measurement above argues it would not pay either, because the gap it is aimed at is not a cache-efficiency gap. |
| A higher-precision recurrence-control projection | Worse on both axes. This file quantizes the Gated DeltaNet control projections `ssm_alpha` and `ssm_beta` to Q4_K, the lowest precision it contains, and `AtomicChat/Qwen3.8-27B-GGUF` carries the same tensors at Q8_0 for 11.4 MiB more. Measured on one machine with one draft head and only the target file changed: mean accepted length 3.22 here against 2.76 there, and 32.90 tok/s against 31.37. The build with the higher-precision projections drafts half a token less per pass. |
| A bfloat16 recurrent state | The ggml operator set requires f32. SGLang serves this model family at `--mamba-ssm-dtype bfloat16`, which halves a state slot from 153.9 to 78.4 MB, and llama.cpp fixes `GGML_TYPE_F32` at every recurrent-memory construction site. Halving the state would halve the per-verification snapshot, about 3.9% of the cycle, and the per-checkpoint cost with it. The type takes six lines to make selectable, after which the first decode aborts on `ggml-cuda/scale.cu:28: GGML_ASSERT(src0->type == GGML_TYPE_F32)`. The Gated DeltaNet graph scales the state and that operator is f32-only, as are others in the chain. |
| DFlash drafting (`--spec-type draft-dflash`) | Ruled out on the publishers' own numbers. The flag is in the binary and DFlash is widely described as beating MTP, so it is the obvious next thing to try, but all three Qwen3.8-27B drafters published for it report acceptance below 3.01 for this MTP head. `mrchuy/Qwen3.8-27B-DFlash-drafter-bootstrap-GGUF` is Qwen3.6 weights transplanted onto the Qwen3.8 tokenizer and reaches 2.02 tokens per verification, acceptance 0.572 / 0.297 / 0.152; an A/B by its author on one target and one machine is 31.94 tok/s for DFlash against 50.70 for native MTP. `kstoyanov99/Qwen3.8-27B-Dflash` is genuinely SpecForge-trained on Qwen3.8 and measures 1.81 accepted tokens per step at 51.27% first-position acceptance. `rwmacy/qwen3.8-27b-dflash-drafter-fp8-b70` claims a mean acceptance length of 2.5 to 3.5, but it ships FP8 safetensors and no GGUF, it was fine-tuned partly on the benchmark domain it reports, and it requires a forked vLLM carrying an off-by-one fix in the DFlash readout without which acceptance collapses to about 24%. A 1.4-1.7 GiB drafter also costs far more per step on this card than the 0.611 GiB head in Table 4. Related: llama.cpp [issue #24541](https://github.com/ggml-org/llama.cpp/issues/24541) reports EAGLE3 against a `qwen3_5` hybrid target running slower than MTP despite acceptable acceptance. |
| Raising the MMVQ warp count at verification width | Worse, and it closes the last cheap route to the MMQ admission fee. The L4 resolves to `MMVQ_PARAMETERS_GENERIC`, whose `calc_nwarps` drops from 4 warps to 2 at `ncols_dst >= 5` while the grid stays fixed by output rows, so verification at width 6 runs at half the warps of width 4. On Ada the 16-blocks-per-SM limit pins 2 warps at 66.7% occupancy where 4 warps reach 100%, which is the shape of a fix for a bandwidth-bound kernel. Built both from one commit and measured on prose, code and json: MMVQ at 2 warps gives 21.02 tok/s and at 4 warps 19.46, against the MMQ 31.38. The occupancy gain is real and is beaten by what it costs, because at 4 warps the K loop wastes more of its last trip (68 of 96 block-slots on the K=17408 matrices against 68 of 80 at 2 warps). The useful part of the result is the margin: MMVQ is 10 tok/s behind MMQ at this width, so no warp-count tuning reaches it and the admission fee needs a different kernel, not a better-configured one. |
| Locking SM clocks | Nothing, across the whole usable range. At a fixed 72 W cap, auto / 2040 / 1800 / 1600 / 1400 / 1200 MHz give 31.14 / 31.41 / 31.22 / 31.29 / 31.35 / 31.27 tok/s, and only 1000 MHz finally breaks the tie downward at 29.65. The card is pinned at the cap at every clock above 1200 MHz, and no clock setting spends that budget better than the governor does. |
| `-ub` 64 to 512, `-b`, `--threads` 4 to 8, `--flash-attn off`, `--no-op-offload` | All within 0.7% at batch 1. Re-swept on the tuned configuration, `-ub` 16 / 32 / 64 / 128 / 256 gave 29.92 / 28.70 / 29.87 / 29.15 / 29.16, a 1.2 tok/s spread with no trend. `-ub 512` is kept because it helps prefill, which decode does not care about. |
| `GGML_CUDA_GRAPH_OPT=1` | 29.33 against 29.36. Its fusion pattern requires a three-way fan-out at batch 1, which only 16 of the 64 blocks in this model have. |
| `GGML_CUDA_DISABLE_FUSION=1` (control) | 28.45 against 29.36, so the existing fusions are worth about 3% and should be left on. |
| llama.cpp master (commit `01818e4`) against the pinned b10454 | 23.38 against 23.60, no difference. |
| Draft head output tensor at Q3_K | 28.89 against the Q4_0 29.20 and the Q2_K 29.91. Acceptance fell faster than bandwidth. |
| Raising `--spec-draft-p-min` with the cheap head | Monotonically worse: 28.71 at 0.0 down to 25.13 at 0.6. |
| PR [#26705](https://github.com/ggml-org/llama.cpp/pull/26705), branchless Q4_K/Q5_K scale unpack | Nothing, and it cannot help here. It removes a per-column re-execution of the scale unpack inside `mul_mat_vec_q`, but this configuration routes verification to `mul_mat_q`. Built and measured: with the patch, MMVQ verification still runs at 23.4 tok/s at depth 3 and 21.9 at depth 5, against 29.6 for MMQ. |
| `DimInfer/Qwen3.8-27B-Dspark-v1`, requantized to Q4_K, at the depth its author recommends | 28.88 / 28.92 / 28.62 at n-max 4 / 5 / 6 with `p-min 0`. Its published mean accepted length of 3.807 at n-max 6 does not reproduce on this workload mix: measured 2.65 / 2.74 / 2.77, against 3.01 for the native MTP head. Their published figures are dominated by math and gsm8k, which speculate far better than chat or json. |
| `xkm/qwen3.8-27b-mtp-head-retrained`, a retrained nextn head | Worse at equal cost. Swapped into the sidecar and requantized to match, it gives mean accepted length 2.85 against 2.89 for the stock head. Its published gain is real but requires an F16 head, and on a card this bandwidth-starved the extra 682 MiB per draft step costs more than the acceptance buys. The same artifact wins on Apple Silicon, where the author measured it. |
| An importance matrix for the draft head | Worth ~1% of accepted length and nothing measurable in throughput. Neither `mtp-Qwen3.8-27B-Q4_0.gguf` nor its Q2_K requantization carries imatrix metadata, and Q2_K is the most imatrix-sensitive quantization available. Computing one (120 chunks of neutral public text, final PPL 3.86) and requantizing lifts accepted length 2.95 to 2.98 and first-position acceptance 0.760 to 0.772, but the file grows 0.4% and the two cancel: 31.83 against 31.84. |
| The retrained nextn head on a matched quantization grid | Worse across every grid. MLX affine g64 asymmetric requantized to GGUF Q4_0 g32 symmetric is the widest possible grid mismatch, so the same head was rebuilt at Q4_K with the imatrix above, which is the near match: 31.27 and accepted length 2.94, against 31.84 and 2.98 for the stock head. Three grids now (Q4_0, Q4_K with imatrix, F16) all land the same way. |
| n-gram drafting at short match lengths | Still never pays. Retested with `ngram-simple` at match length 4 and 6 and `ngram-mod` at 8, on top of the chain configuration: 30.98, 31.43 and 31.58 against a 31.63 control. Short matches fire often enough to preempt MTP (n-gram has fixed dispatch priority) without drafting anything better. |
| Host and OS tuning, as a bundle and individually | Noise, and measured. Stopping docker, containerd, snapd, packagekit, unattended-upgrades, the Google agents, rsyslog and multipathd, plus transparent hugepages set to `always` and `vm.swappiness=0`, gives 31.20 against a 31.65 control; hugepages alone 31.23; the bundle repeated 31.26. `--threads` 8 / 4 / 2 gives 31.11 / 31.27 / 31.25, and `--prio 2` 31.24. All inside a ±0.5 spread. This is what a fully GPU-resident decode should look like: the host does about a millisecond of work per 33 ms token, so there is nothing there to win. Note also that the g2 shape exposes no cpufreq governor and no C-states, so those knobs do not exist to turn. |
| Drafting deeper than 5 | Accepted length rises and throughput does not. At `p-min 0` the depths 5 / 6 / 7 / 8 / 9 give accepted length 3.22 / 3.26 / 3.49 / 3.38 / 3.38 for 32.63 / 31.90 / 31.41 / 29.29 / 28.75 tok/s. Depth 7 drafts 8.4% more accepted tokens per pass and still loses, because each extra draft step is a full draft-model forward and the verification batch widens past the point where the kernel routing below stays free. |
| Kernel routing held fixed across draft depth | The per-quant-type routing in Table 4 is tuned for one verification width and does not survive a change of depth. MMQ tiles 8 columns at a time, so widths 2 to 8 cost the same on the tensors routed to it, while MMVQ cost grows per column, and `GGML_MMVQ_MAX_Q6K=8` with `GGML_MMVQ_MAX_IQ4XS=8` puts 23% of the weights on MMVQ. At depth 7 the verification batch is 8 wide and that 23% is paying linearly for it: forcing both types onto MMQ recovers 31.41 to 32.52 and lifts the worst of the seven workloads from 24.84 to 26.42. The crossover runs the other way at depth 5, where the same all-MMQ setting gives 32.37 against 32.63. Neither combination beats the shipped one, and the run-to-run spread on this benchmark is about 0.3 tok/s, so depth 7 on MMQ is a tie rather than a win. |
| `--spec-draft-p-min` above 0 with the native head | Worse at every depth tried. At depth 7 on MMQ, `p-min` 0.0 / 0.05 / 0.10 / 0.20 gives 32.52 / 32.36 / 29.92 / 28.29, and at depth 5 `p-min 0.10` gives 31.27 against 32.63. Stopping a draft early saves less than the drafts it abandons were worth. |
| Giving the block drafter its own output head | Worse, and the reason separates it from the native head. The drafter ships no `output.weight`, so `src/models/dflash.cpp` leaves `output` null and every draft round re-reads the target `Q6_K` head, 994.6 MiB. Grafting the `q2_K` full-vocabulary head from the sidecar onto it, which saves 596.7 MiB per round and needs no `d2t` because the vocabulary is unreduced, gives 35.34 tok/s against 35.88 and accepted length 3.40 against 3.52. The same trick raised accepted length on the native head, where the draft output only has to pick one token. This drafter runs a top-k over those logits and traces a candidate path through them, so the logits are structural and `q2_K` noise costs more ranking accuracy than the bandwidth buys. |
| Per-quant kernel routing under the block drafter | Worse both ways, and by more than under the native head. At verification width 8 with the target output head read twice per cycle, all types on MMQ gives 36.01 and 35.93 against 34.94 for `Q6_K` on MMVQ and 34.86 for `IQ4_XS` on MMVQ. The routing in Table 4 is tuned for a narrower width and does not survive the move. |
| `--spec-draft-n-max` above 7 with the block drafter | Nothing, and it is not free. `dflash.block_size` is 8 and `n_draft_max` is `block_size - 1`, so the value is clamped to 7 with a warning. The clamp mutates a copy, while `need_n_rs_seq()` sizes the target recurrent state from the raw value, so `n-max 8` and `n-max 9` draft exactly 7 tokens and spend 299 and 598 MiB of VRAM for it. |
| `--spec-draft-p-min` with the block drafter | Inert. The DFlash 2 branch returns before the confidence gate is reached, so the value is never read. The block is drafted in full every round. |
| Two resident stream-k blocks per SM | Monotonically worse. When tiling efficiency is under 90% the launcher sizes the grid at exactly `nsm`, one block per SM, which is 58 blocks of 8 warps on this card and 16.7% occupancy, and shared memory at ~40 KiB against Ada's 100 KiB per SM leaves room for a second block. Making the multiplier a runtime knob and sweeping it gives 38.29 tok/s at one block, 37.11 at two, 35.87 at three and 35.69 at four. Two blocks does lift the worst workload from 29.31 to 31.36, so the occupancy gain is real; it is simply beaten by what the finer K-split costs everywhere else. This closes the cheap half of the MMQ admission fee, and the expensive half needs a different kernel rather than a different launch. |
| Returning any type to MMVQ on the rebuilt weights | Worse, and the margin grew. The type mix moved by more than two orders of magnitude on some types, so the routing decision was re-taken rather than assumed: all types on MMQ gives 37.74 against 34.37 for `Q6_K` and `IQ4_XS` on MMVQ, and 24.19 with every type on MMVQ. This also retires PR [#26705](https://github.com/ggml-org/llama.cpp/pull/26705), which accelerates the MMVQ unpack by up to 21.7% on this architecture: MMVQ at verification width 8 costs 132.0 ms against MMQ's 90.5, and the patch is inert code under this routing. |
| `GGML_CUDA_DISABLE_GRAPHS=1`, `--cache-ram 0`, `--ctx-checkpoints 0` | Controls. Disabling CUDA graphs costs 1.8% (32.02 against 32.61 on the shipped configuration, and 2.3% on an earlier one), with mean accepted length unchanged at 3.22, so graph capture is already working, and neither the prompt cache nor context checkpointing costs anything measurable. |

Forward-pass cost must be measured through speculative decoding. Timing prompt processing instead shows a 3.3x step between width 4 and width 5 that does not exist in the decode path, because prompt processing and speculative verification take different paths through the server.

## Memory and context

The draft head is a second model resident in VRAM, so it is paid for in context. Table 13 reports VRAM with the server loaded and idle at an f16 KV cache, on a 24,089 MiB device, which is the measurement the shipped configuration is planned against.

**Table 13: VRAM against context size, f16 KV cache.**

| `--ctx-size` | VRAM used | Free |
| --- | --- | --- |
| 8,192 | 18,960 MiB | 5,129 MiB |
| 32,768 | 20,616 MiB | 3,473 MiB |
| 65,536 | 22,848 MiB | 1,722 MiB |
| 81,920 | 23,946 MiB | 143 MiB |
| 90,112 | fails to load | |

The free column is the operative one. At 81,920 the server loads, serves and benchmarks at full speed with 143 MiB free, and then aborts on the first request whose batch shape has not been seen before, because llama.cpp instantiates a CUDA graph per distinct shape and instantiation allocates:

```
ggml-cuda.cu:106: CUDA error
CUDA error: out of memory
  cudaGraphInstantiate(&graph->instance, graph->graph, __null, __null, 0)
```

A fixed benchmark replays a fixed set of shapes and never reaches this. The 40-prompt accuracy check, with `enable_thinking` off and 40 distinct short prompts, reaches it immediately. A context size is therefore validated against varied prompt shapes, and the memory it leaves free matters as much as the memory it uses.

### Context ceiling

Sixteen of the 64 layers keep a KV cache. The other 48 are Gated DeltaNet and carry a fixed recurrent state that does not grow with context. Those 16 layers cost 64 KiB per token at f16, so the cache type sets how much context a given amount of memory buys.

The cache is not the only claim on that memory. llama.cpp allocates a compute graph buffer inside `llama_context::decode`, sized by batch shape, after the model and the cache are already resident. A context size chosen by loading the server and exercising it can therefore serve correctly and still exhaust the device later, when a batch shape that was never exercised needs a larger buffer:

```
llama_context::decode -> process_ubatch -> ggml_backend_sched_alloc_graph -> ggml_gallocr_alloc_graph
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 158.66 MiB on device 0: cudaMalloc failed: out of memory
```

llama.cpp sizes this on its own. `--fit`, on by default, adjusts unset arguments to fit device memory, and `--fit-target` sets the margin it leaves, 1024 MiB per device by default. Setting both `-ngl` and `--ctx-size` disables the planner, which reports that at every start:

```
common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort
```

Leaving them unset yields about 26 tokens of context for every MiB of margin surrendered.

**Table 14: context granted by `--fit` at each margin, with `-ub 512` and a q8_0 KV cache.**

| `--fit-target` | Granted ctx | VRAM after load |
| --- | --- | --- |
| 1024 MiB | 68,096 | 22,220 MiB |
| 768 MiB | 75,008 | 22,484 MiB |
| 512 MiB | 81,664 | 22,736 MiB |
| 256 MiB | 88,320 | 22,990 MiB |

The margin is what a context costs to hold. A smaller micro-batch and a cheaper cache type buy the window back while keeping it:

**Table 15: context granted at a 768 MiB margin.**

| `-ub` | KV | Granted ctx |
| --- | --- | --- |
| 512 | q8_0 | 75,008 |
| 256 | q8_0 | 81,664 |
| 128 | q8_0 | 84,992 |
| **512** | **q4_0** | **127,488** |
| 256 | q4_0 | 139,776 |
| 128 | q4_0 | 146,176 |

q4_0 KV at `-ub 512` is granted 127,488 tokens at a 768 MiB margin. Accuracy is 40/40 at f16, q8_0 and q4_0 alike, the weights are `UD-Q4_K_XL` in every case, and the cache type is an independent choice, the one the sibling [Qwen3.6-35B-A3B box](https://github.com/hanxiao/Qwen3.6-35B-A3B-MTP-L4) already serves at q4_0.

That grant is not the usable window, because the planner sizes what it can see at load and the device keeps filling afterwards. Sampled against context depth, VRAM rises **12.25 KiB for every token resident**, linearly and independently of batch shape or checkpoint count. Filling 127,488 tokens therefore costs about 1.5 GiB beyond the load-time footprint, against the 2.0 GiB the 768 MiB margin leaves free, and a long conversation exhausts the device before it reaches its own advertised ceiling. The failure surfaces as a lazily allocated compute buffer:

```
llama_context::decode -> process_ubatch -> ggml_gallocr_alloc_graph
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 258.69 MiB on device 0: cudaMalloc failed
```

Sizing for that growth rather than against it gives the shipped configuration: `--fit-target 1792` for 81,664 tokens, which leaves about 2.1 GiB free once the window is full. Filled to depth 77,023 with prompt lengths varied across turns, it holds that margin and answers a request past the ceiling with HTTP 400 instead of aborting. `--ctx-checkpoints 4 --checkpoint-min-step 16384` bounds the recurrent-state copies a long conversation accumulates.

**Table 16: what a granted context leaves free once it is full.**

| `--fit-target` | Granted ctx | Free at full depth |
| --- | --- | --- |
| 768 MiB | 127,488 | 555 MiB |
| 1280 MiB | 104,550 | 1,344 MiB |
| **1792 MiB** | **81,664** | **2,134 MiB** |

A client that needs the window should read it from `/props`. A second copy of the number drifts from the first.

If the priority is instead the largest window with no KV quantization at all, drop `--spec-draft-model` and keep the kernel patch: that frees the 611 MiB of the draft head, measures 28.05 tok/s and fits the original 104,192.

## Quality

`scripts/verify-quality.sh` runs two independent checks against a `--spec-type none` reference: byte-level determinism and accuracy on 40 arithmetic problems with known ground truth. It captures three configurations so the kernel change and speculation can be told apart.

**Table 17: quality checks.**

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

The build checks out the head of [PR #27173](https://github.com/ggml-org/llama.cpp/pull/27173) (one-decode chain drafting, the `LLAMA_SPEC_CHAIN` feature this configuration depends on), applies `patches/0001-mmvq-runtime-crossover.patch`, and targets SM89 only. Set `PR=` and `SHA=<commit>` to pin a plain commit instead. On 32 vCPUs it takes about four minutes; on the 8 vCPUs of a `g2-standard-8` allow twenty. Building on a separate CPU instance and copying `bin/` to `/opt/llama.cpp/build/bin` is faster and keeps the benchmark machine idle.

The patch is one predicate in `ggml/src/ggml-cuda/mmvq.cu` plus a log-level change in `common/speculative.cpp` that promotes the speculative statistics in llama.cpp (mean accepted length, per-position acceptance, cumulative drafting time) from trace to info. Every number in the cost model above came from that line; without it the only way to see drafting cost is `-lv 6`, which logs every draft candidate and slows generation by 30%.

## Network access

The server binds `0.0.0.0` and has no authentication, so the firewall is the only thing standing between the model and the internet. The provisioning script defaults to `INTERNAL=1`, which tags the instance `llama-internal` and admits `tcp:8080` from RFC1918 only. `INTERNAL=0` opens it to `0.0.0.0/0` through the `llama-server` tag, which is convenient for a throwaway benchmark and wrong for anything that stays up.

If an instance was created with `INTERNAL=0` and you want to close it later, move the tag instead of adding a rule:

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

Table 18 lists the files available in [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). Files prefixed `UD-` use Unsloth Dynamic v3.0. Only `UD-Q4_K_XL` is measured here; the fit column compares file size against device capacity.

**Table 18: quantizations and fit on a 24 GB L4.**

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
  --metadata=qwen-hf-file=Qwen3.8-27B-UD-Q5_K_XL.gguf,qwen-fit-target=1024
```

The draft head is tied to the model family, not to the quantization of the target, so it does not change with the target file.


## Cost

An on-demand `g2-standard-8` costs approximately $0.81/hr and bills while the instance is RUNNING, independent of load. At 32.5 tok/s that is 144,000 tokens per dollar, against 107,000 before.

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

A stopped instance restarts with `bash scripts/start.sh`, keeping its disk, the models and the build, but the zone does not hold capacity for it. Restarts return `STOCKOUT` exactly as creation does, and a stopped instance cannot change zones, so the script retries in place until an L4 frees up. Serving flags come from instance metadata, so a restart picks up whatever the metadata currently says. `startup.sh` reads `qwen-fit-target`, `qwen-kv`, `qwen-nmax`, `qwen-mmvq-max`, `qwen-chain-sub` and `qwen-lcpp-pr` for serving and build behaviour, and `qwen-hf-repo` / `qwen-hf-file` / `qwen-draft-repo` / `qwen-draft-file` to choose the model and draft head; all have defaults, and `provision-ondemand.sh` sets only the first three.

When `/health` never comes up, the boot log carries the reason:

```bash
gcloud compute ssh qwen38-27b-l4-od --zone=$ZONE --command 'sudo tail -80 /var/log/qwen-startup.log'
```

## Draft head reference implementation

The throughput ceiling section shows the one remaining lever is a draft head with a higher accepted
length, and that is a training project. Training needs the head to run outside llama.cpp.
`tools/mtp_head_reference.py` is that forward pass, and it scores **68.8% top-1** on tok_{i+2} given
(h_i, tok_{i+1}), against 0.774 acceptance at draft position 0 in production.

Two conventions in this checkpoint are not visible from the HF files, and both are silent:

| Convention | Detail |
| --- | --- |
| RMSNorm weights are offsets | The effective scale is `1 + w`. `convert_hf_to_gguf.py` bakes the +1 in, so the GGUF tensor equals the HF tensor plus 1.0 exactly on all seven norms of this block |
| `q_proj` packs two things | Q and the attention gate are interleaved per head at `head_dim * 2` stride |

The first one is not a subtle numerical difference. `mtp.pre_fc_norm_embedding` is 100% negative in
the HF checkpoint at mean -0.4606, and `mtp.pre_fc_norm_hidden` is 96.1% negative, so a port that
uses `w` where the model means `1 + w` negates both block inputs at the first operation. The measured
result is 0.0% top-1 with the true token at median rank 247,841 of 248,320, anti-correlated rather
than scattered. Applying the convention moves that to 68.8% with no other change.

Nothing upstream catches this. transformers declares
`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`, so these tensors are never loaded there and no
reference implementation exercises them.

The diagnostic that isolated it is worth reusing on any ported block: trace
`cos(stage_output, h)` at every stage. The sign flip appears at the first norm, where
`cos(h_norm, h)` is -0.899, which points at a scale convention rather than at graph structure.

| Setting | Value | Margin over the alternative |
| --- | --- | --- |
| RoPE style | rotate_half | 68.8% against 15.8% |
| Q and gate split | interleaved | 68.8% against 16.7% |
| `rope_theta` | 10,000,000 | |
| Rotary dims | 64 of 256, `partial_rotary_factor` 0.25 | |
| `mrope_section` | `[11, 11, 10]`, 32 pairs, equal to standard RoPE for text-only input | |

Top-1 is flat at 66 to 69% across causal warmup depths of 32 through 1,536, so the gap to the
production figure is workload mix, not truncated context.

## Draft head training

The throughput ceiling section identifies accepted length as the one lever left, so the head was
trained. `tools/train_nextn_head.py` does it: it reads captured target hidden states, and the label
is the target argmax at the next position, because acceptance is the head emitting what the target
would emit. Restricting the loss to the captured top-8 ids removes the 248,320-wide `lm_head` from
the backward pass, which brings one epoch over 3M positions down to seven minutes on the L4.

Training works, and it does not transfer. Four heads were trained on a 9,324-document, 4.5M-token
calibration corpus and each was measured on the seven-workload bench, which the training corpus does
not overlap.

| Head | Held-out top-1, training corpus | Bench tok/s | Accepted length |
| --- | --- | --- | --- |
| Qwen released head | 50.20% | 32.61 | 3.22 |
| Fine-tuned, lr 3e-6, 292 steps | 64.05% | 31.76 | 3.18 |
| Fine-tuned, lr 2e-5, anchored to the released weights | 71.83% | 30.38 | 3.02 |
| Fine-tuned, lr 2e-5, 1,464 steps | 72.31% | 29.49 | 2.89 |

The relationship is monotone: every point gained on the training corpus costs throughput on the
bench. Learning rate and an anchor penalty move a head along that line rather than off it, so
neither is the lever. The corpus is. A head trained this way generalizes across documents inside its
own corpus, gaining 22 points on a disjoint tail, and still loses 0.33 of accepted length everywhere
else, which is the signature of domain shift rather than memorization.

One structural gap remains open and is implemented but unmeasured. Teacher forcing only ever shows
the head the target hidden state, while `common/speculative.cpp` feeds the head its own pre-norm
state forward through `llama_get_embeddings_nextn_ith` for draft positions 1 and beyond. Measured
per-position acceptance is 0.78, 0.55, 0.35, 0.25, 0.18, so almost all of the accepted length is
lost in that decay, and teacher forcing never trains it. The trainer unrolls K steps and feeds the
pre-norm state forward to match the drafter. Feeding the true token at each step is correct, because
acceptance at position k is conditional on positions 0 through k-1 already matching.

Beating a head the model authors trained needs training data at least as broad as theirs. That is
the constraint on this path, and it is a data problem, not a method or a serving problem.

## Adaptive draft depth

A single draft depth gives up one workload class to serve another. Measured on this card at fixed
depth, on GSM8K-shaped prompts depth 7 reaches 46.9 tok/s at mean accepted length 4.45 while depth 5
reaches 43.0 at 3.93, and on the seven-workload benchmark that ordering reverses. The seven-workload
mean and the worst workload in it disagree the same way, so the depth that maximizes one loses the
other.

`patches/0002-adaptive-draft-depth.patch` lets the depth follow the workload. The control signal is
whether the draft was **fully consumed**, not how many tokens were accepted:

| Observation | Meaning | Action |
| --- | --- | --- |
| `n_accepted >= n_drafted` | the draft ran out before the target disagreed | depth + 1, capped at `LLAMA_ADAPT_HI` |
| `n_accepted + slack < n_drafted` | the target rejected well short of the draft | depth - 1, floored at `LLAMA_ADAPT_LO` |
| otherwise | the depth is where the workload supports it | hold |

An exponential moving average of the accepted count does not work, and the failure is instructive. It
is a positive feedback loop: accepting more tokens raises the estimate, which drafts deeper, which
accepts more, so every workload with any acceptance runs to the cap. Measured cost of that version:
the math workload drafted 391 tokens against 308 for an identical 191 accepted, and lost 5.7%. Full
consumption is the only evidence that a deeper draft would have paid, which makes the controller
self-limiting.

| Workload | Fixed depth 5 (tok/s) | Adaptive 5 to 7 (tok/s) |
| --- | --- | --- |
| prose | 34.94 | 33.29 |
| code | 24.54 | 29.76 |
| json | 32.95 | 32.02 |
| chat | 26.74 | 27.17 |
| math | 43.19 | 42.86 |
| multi | 32.02 | 32.65 |
| summ | 33.65 | 33.89 |
| **mean** | **32.58** | **33.09** |
| **worst** | **24.54** | **27.17** |
| GSM8K-shaped | 42.83 | 45.79 |

Alternated against the fixed-depth configuration at four repetitions each, the fixed arm returns
32.62 and 32.54 and the adaptive arm returns 33.12 and 33.12, so the 1.7% is outside the run-to-run
spread. The worst workload gains 10.4% and GSM8K-shaped prompts 6.9%. Accuracy is 40/40.

Adaptive depth also reverses the per-quant kernel routing. That routing is tuned for one verification
width, and once the width moves the tuning moves with it: at depth 5 returning `Q6_K` and `IQ4_XS` to
MMVQ is worth 9% on the math workload, and under adaptive depth the same setting costs 21% on the
code workload. Holding all types on MMQ is worth more than the routing it gives up, so the two
settings ship together.

## A block-diffusion drafter

The ceiling analysis says accepted length is the only lever with enough leverage left, and that a
better drafter is a training project. Someone else did that training.
[DFlash 2](https://inco.ai/blog/dflash2/) is a block-diffusion drafter published for this exact
target: it predicts a whole block in one pass rather than one token per pass, keeps candidates at
every position, and traces a path through them with a small selector. Decoding is lossless.

Two properties matter on this card. One pass per block replaces the five to seven sequential draft
forwards that cost 8.3 ms of a 98.8 ms cycle, and the accepted length is far higher than the native
nextn head reaches. Both act on the two terms the cost model says are worth attacking.

| Drafter, both on the same binary | Seven workloads | Worst workload | GSM8K-shaped accepted length |
| --- | --- | --- | --- |
| Native nextn head, depth 5 | 31.30 tok/s | 23.31 | 4.123 |
| DFlash 2 `Q4_K_M`, `n-max 7` | **35.99 tok/s** | 28.12 | **5.573** |

Against the configuration this repository previously measured as best, every workload gains except
json:

| Workload | Native head, depth 5 | DFlash 2, `n-max 7` |
| --- | --- | --- |
| prose | 34.94 | 36.83 |
| code | 24.54 | 31.13 |
| json | 32.95 | 32.09 |
| chat | 26.74 | 28.12 |
| math | 43.19 | 48.14 |
| multi | 32.02 | 39.75 |
| summ | 33.65 | 35.83 |
| **mean** | **32.58** | **35.99** |
| **worst** | **24.54** | **28.12** |
| GSM8K-shaped | 42.83 | 55.34 |

Three independent runs give 35.92, 36.01 and 35.99, against a run-to-run spread of about 0.4, and
accuracy is 40/40. The published accepted length for this drafter is 5.39 on the first eight GSM8K
examples against a `Q4_K_M` target; against the `UD-Q4_K_XL` target here it measures 5.573, so it
transfers rather than depending on the target quantization it was evaluated on.

Settings, each measured rather than assumed:

| Setting | Value | Against the alternative |
| --- | --- | --- |
| Draft quantization | `Q4_K_M`, 1.14 GiB | 35.99 against 35.39 for `Q8_0`, which agrees with the publisher's own ordering |
| `--spec-draft-n-max` | 7 | 35.99 against 34.76 at 5 and 35.86 at 8. The block caps at 7, so 9 returns the identical accepted length |
| Kernel routing | all types on MMQ | 36.01 against 34.57 for the per-quant routing, which is tuned for a narrower verification width |
| `-ub` | 512 | 35.89 at 256, no difference |

It needs [PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342), which is a different branch
from the chain-drafting PR the native-head configuration is built on. Chain drafting is a
one-decode optimization for sequential drafting, so a block drafter has nothing to gain from it, and
the two do not need to be combined. `patches/0001-mmvq-runtime-crossover.patch` applies to that
branch unchanged.

## Long-horizon behaviour and the context ceiling

The seven-workload benchmark runs 256-token generations from short prompts. The deployment does not:
it runs one growing conversation at 80k or more tokens with heavy prefix reuse. Those two regimes
disagree, and the block drafter changes the answer on both.

Measured by growing a single conversation turn by turn, each turn reusing the whole prefix, at the
window the memory planner grants for a 1,024 MiB margin:

| Turn | Context | Decode tok/s | Accepted length | Prefill tok/s | Prefix cached | VRAM MiB |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 21,963 | 34.12 | 3.76 | 554.5 | 0% | 23,786 |
| 2 | 43,696 | 33.13 | 4.09 | 414.2 | 50.0% | 23,786 |
| 3 | 65,429 | 27.93 | 3.76 | 334.4 | 66.7% | 23,786 |
| 4 | 87,162 | 29.76 | 4.36 | 279.9 | 75.0% | 23,786 |
| 5 | 108,895 | 29.10 | 4.57 | 240.5 | 80.0% | 23,786 |
| 6 | 130,628 | 26.35 | 4.47 | 210.9 | 83.3% | 23,786 |
| 7 | 152,361 | 22.17 | 4.00 | 187.3 | 85.7% | 23,786 |
| 8 | 174,094 | 20.29 | 3.92 | 168.7 | 87.5% | 23,786 |

Two things in that table are worth more than the throughput column. Accepted length **rises** with
depth, from 3.76 to 4.57, so long-horizon work speculates better than the benchmark suggests and the
drafter is worth more in the deployment than on the bench. And VRAM is **flat** at every depth. The
configuration this repository previously served grew 12.25 KiB per token of depth and aborted on a
graph buffer allocation; this one does not move. A twenty-case shape battery, walking prompt lengths
across many final-ubatch remainders, passes 20 of 20 at the same window, and the server log records
zero allocation or graph failures across the whole run. Turn 9 is refused with a 400 because it would
exceed the window, which is the correct behaviour rather than a failure.

The context gain is the drafter, not the build. Measured on one binary, changing only the drafter:

| Drafter, `--fit-target 1792`, `-ub 512` | Granted context | VRAM MiB |
| --- | --- | --- |
| Native nextn head, `n-max 5` | 81,664 | 21,422 |
| DFlash 2, `n-max 7` | 155,136 | 22,798 |

The native head reproduces exactly the window this repository served before, so the comparison is
clean. Its draft context carries a KV cache that scales with the target context; the block drafter's
does not, and that is where the window comes from.

### Sweeping the ceiling

The planner grants a window from the margin it is told to leave. Granted is not the same as usable,
so every candidate below was put through a twenty-case shape battery, which walks prompt lengths
across many final-ubatch remainders, and then a deep fill that grows one conversation toward the
ceiling. Both are necessary. The 249,600 window loads cleanly, reports itself healthy, and then fails
every one of the twenty shapes with a hundred allocation failures.

**Table 19: granted and usable context, on the rebuilt weights with the block drafter.**

| `--fit-target` | `-ub` | Granted | Free at idle (MiB) | Shape battery | Deep fill | Allocation failures |
| --- | --- | --- | --- | --- | --- | --- |
| 1792 | 512 | 173,568 | 1,768 | 20/20 | 152,361 | 0 |
| 1536 | 512 | 185,088 | 1,508 | | | |
| 1280 | 512 | 196,352 | 1,256 | | | |
| 1024 | 512 | 207,872 | 998 | | | |
| 768 | 512 | 219,392 | 738 | 20/20 | 185,485 | 0 |
| 640 | 512 | 225,024 | 610 | **2/20** | | 2 |
| 512 | 512 | fails to load | | | | |
| 1792 | 256 | 179,456 | 2,036 | | | |
| 1024 | 256 | 214,528 | 1,266 | | | |
| **768** | **256** | **226,048** | **1,014** | **20/20** | **222,536** | **0** |
| 512 | 256 | 237,824 | 754 | 20/20 | 222,536 | 0 |
| 256 | 256 | 249,600 | 496 | **0/20** | fails turn 1 | 100 |

`-ub 256` is what makes the large windows usable, not the margin. At `-ub 512` the ceiling breaks
between 219,392 and 225,024, while at `-ub 256` it survives to 237,824 on a smaller margin, because
the compute buffers a graph instantiation has to find room for are half the size. It costs 1.7% of
prefill and nothing measurable in decode.

The shipped window is **226,048** at `--fit-target 768 -ub 256`. 237,824 passes the same checks and
is 5% larger, and it is not taken: it leaves 754 MiB at idle and 564 MiB during a deep prefill,
against 1,014 and 824 for the shipped one, and both reached the same depth. A context ceiling on this
card has aborted production twice, and the margin is what the second abort was spent buying back.

For reference at `q8_0` KV, which is a different axis from the weight floor: 102,144 at a 1792
margin and 129,024 at 768. `q4_0` KV is the better trade on both throughput and window.

## The 2026-08-19 rebuild of the weights

Unsloth re-cut `UD-Q4_K_XL` under Dynamic 3.0 on 2026-08-19. It is not the same file: 306 of its 866
tensors changed type, and it is 364 MB smaller while spending more precision where the architecture
needs it.

| Type | Old file | New file |
| --- | --- | --- |
| Q8_0 | 0 | 110 |
| Q6_K | 19 | 56 |
| Q5_K | 325 | 191 |
| Q4_K | 97 | 69 |
| IQ4_XS | 65 | 70 |
| IQ4_NL, Q3_K, IQ3_S | 0 | 10 |

The largest single move is the recurrence control projections: 48 `ssm_alpha` and 48 `ssm_beta`
tensors go `Q4_K` to `Q8_0`. Those are the tensors the earlier comparison against a third-party build
identified as the ones this architecture is sensitive to, and the model authors have now raised them
in the reference file.

Measured, changing only the file:

| | Old file | New file |
| --- | --- | --- |
| Seven workloads | 36.62 tok/s | **37.86** |
| json workload | 32.09 | **40.37** |
| Context at `--fit-target 1792` | 155,136 | **173,568** |
| GSM8K-shaped accepted length | 5.559 | 5.399 |

It is worth 3.4% of throughput and 18,432 tokens of context at once, and it repairs the one workload
where the block drafter had been losing. It drafts very slightly worse, which is consistent with the
recurrence projections moving, and the benchmark mean prefers it by far more than the drafting costs.
Accuracy is 40/40 on the new file.

Pin the size. The old file is 17,923,394,624 bytes and the new one is 17,559,178,144, and both are
served under the same name.

## Files

| Path | Purpose |
| --- | --- |
| `scripts/provision-ondemand.sh` | create the L4, then hand off to `startup.sh`, and wait for health |
| `scripts/startup.sh` | on-instance: ECC off, build llama.cpp, fetch both models, install the systemd unit |
| `scripts/build-llamacpp.sh` | build a patched, SM89-only llama.cpp anywhere with a CUDA toolkit |
| `patches/0001-mmvq-runtime-crossover.patch` | adds `GGML_MMVQ_MAX`, the kernel crossover knob |
| `patches/0002-adaptive-draft-depth.patch` | draft depth follows the workload, gated by `LLAMA_ADAPTIVE_DRAFT` |
| `scripts/bench.sh` | seven-workload decode benchmark |
| `scripts/sweep-spec.sh` | speculative decoding parameter sweep; still drives the previous docker path, so point it at `/opt/llama.cpp/build/bin/llama-server` before use |
| `scripts/max-context.sh` | binary search for the largest usable context; same caveat |
| `scripts/verify-quality.sh` | determinism and accuracy checks |
| `tools/bandwidth-probe.cu` | measure achievable DRAM bandwidth; used to separate the power tax from kernel slack |
| `tools/capture-mtp-data.cpp` | capture MTP draft-head training data at prefill speed, for the drafter project the ceiling analysis points to |
| `tools/mtp_head_reference.py` | torch forward pass for the nextn draft head, matching `src/models/qwen35.cpp` |
| `tools/train_nextn_head.py` | fine-tune the draft head on captured hidden states, unrolled K steps |
| `scripts/start.sh` | restart a stopped instance, retrying through STOCKOUT |
| `scripts/teardown.sh` | stop or delete |

`capture-mtp-data.cpp` emits exact, self-consistent records: the argmax equals the top-1 id in all of them, the hidden states are fully dense at 5,120 nonzero components, and there is no ragged tail. Each position stores a bf16 hidden state, so a record is **10,312 bytes**, of which the top-K logits are only 64. That is 10.3 GB per million tokens, which is why `provision-ondemand.sh` takes a `DISK` knob. A capture of 5,056,189 positions occupies 48.56 GB and takes 2h51m of L4 time.

## Related

[hanxiao/Qwen3.6-35B-A3B-MTP-L4](https://github.com/hanxiao/Qwen3.6-35B-A3B-MTP-L4) serves a sparse MoE on the same card at 92-100 tok/s.

## License

The scripts in this repository are Apache-2.0. The model weights are covered by their own licence: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and the [Unsloth GGUF conversion](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) are both Apache-2.0. The draft head comes from [ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF), same licence.
