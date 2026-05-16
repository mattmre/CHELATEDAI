# Disk-Resident LLM Feasibility For ChelatedAI

Date: 2026-03-28

## Scope And Assumption

The user request referenced "this paper" without attaching the paper in the session context.
For this memo, the working assumption is that the target paper is **LLM in a Flash** because it is the closest primary-source match to the repo's disk-first computational-storage direction:

- weights stored on flash / SSD
- selective loading into host memory
- explicit avoidance of full GPU-resident execution

To sharpen the CPU side of the design, this memo also uses:

- **T-MAC** for lookup-table CPU kernels on low-bit models
- **ReLU Strikes Back** for retraining a model into a more flash-friendly sparse FFN regime
- **SeedLM** as an optional next-stage compression idea when disk bandwidth becomes the main bottleneck

If the intended paper is different, replace the external method assumptions in Sections 3-5 and keep the repo-side observations unchanged.

## Short Answer

ChelatedAI can plausibly evolve into a **disk-resident, CPU-executed inference path** for quantized sparse models.
It cannot support that claim today.

The current repo proves:

- toy and digits-class block-graph replay correctness
- deterministic transport semantics
- theoretical latency modeling

The current repo does **not** prove:

- a transformer running from disk
- realistic SSD bandwidth behavior
- a CPU lookup kernel for low-bit inference
- a controller-resident execution path on commodity SSDs

The strongest feasible near-term architecture is:

1. SSD/NVMe stores compressed model weights.
2. DRAM keeps the permanently resident subset plus small rolling caches.
3. CPU executes low-bit kernels directly, ideally LUT-based.
4. SSD is treated as a bandwidth source, not as a general-purpose transformer compute engine.

That is materially more defensible than claiming the SSD controller itself will replace the GPU.

## What The Repo Already Has

Relevant current surfaces:

- `computational_storage_poc/block_graph.py`
- `computational_storage_poc/mock_nvme.py`
- `computational_storage_poc/mock_array.py`
- `computational_storage_poc/README.md`
- `docs/COMPUTATIONAL_STORAGE_DRIVE_NODES.md`
- `docs/computational-storage-transport-scope-decision.md`

Current strengths:

- block-graph serialization and replay already give the repo a storage-native execution abstraction
- `mock_array.py` already frames multi-drive parallelism as a first-class idea
- the scope decision doc correctly prevents over-claiming hardware maturity

Current blockers:

1. `block_graph.py` stores fixed `512 x 512` dense FP16 payloads, zero-padded. That is simple for parity testing, but wasteful for large transformer layers and incompatible with selective sparse loading.
2. `mock_nvme.py` loads the full file into memory at initialization, so it is not exercising real disk behavior.
3. The model assumes dense matrix traversal, while `LLM in a Flash` depends on keeping attention resident and loading only small active FFN slices.
4. There is no real CPU low-bit kernel path. The current mock assumes theoretical compute rates rather than measuring actual CPU kernels.
5. The transport proof is intentionally toy-scoped and should stay that way until a transformer workload exists.

## Best Adaptation Strategy

### Recommended interpretation

Do **not** try to adapt the paper as "drive controller does all inference."

Do adapt it as:

- disk-resident weight store
- host-side DRAM working set
- CPU-side low-bit execution
- sparse selective FFN streaming
- optional multi-drive sharding for bandwidth scaling

That is consistent with both the paper and the repo's current maturity.

### Concrete architecture change

Split the model into three classes of artifacts:

1. **Resident tensors**
   - embeddings
   - attention weights
   - norms
   - routing metadata / predictor weights

2. **Streamed tensors**
   - FFN up/down projections
   - optional MoE experts
   - per-layer sparse bundles

3. **Index / manifest data**
   - per-layer offsets
   - row-column bundles
   - quantization metadata
   - predictor metadata
   - sharding placement for multi-drive layouts

### Why this matches the literature

`LLM in a Flash` explicitly keeps attention resident and streams only a small fraction of FFN weights per token.
`ReLU Strikes Back` argues that retraining with ReLU can materially increase activation sparsity and reduce transferred weights.
`T-MAC` shows that once the model is low-bit, CPU-side LUT kernels become credible instead of dequantizing everything first.

## Quantitative Feasibility

The new in-repo estimator is:

- `computational_storage_poc/disk_llm_estimator.py`

It models two upper bounds:

- **dense_tps<=**: worst-case if each token forces the full model to stream from SSD
- **flash_tps<=**: optimistic `LLM in a Flash`-style upper bound where:
  - one-third of weights stay resident
  - two-thirds are streamable
  - only `2%` of streamed FFN weights are fetched per token
  - effective SSD bandwidth is discounted to `70%` of headline bandwidth

The second number is the only path that makes large disk-hosted models plausible.

### Consumer Gen4 NVMe, 32 GB DRAM

Command:

```powershell
python computational_storage_poc\disk_llm_estimator.py --hardware consumer_gen4
```

Observed output:

```text
Disk-resident LLM feasibility (consumer_gen4)
model      size_gb  resident_gb  req_ram_gb  dense_tps<=  flash_tps<=  fits_ram  fits_2x_dram
7B/4b          3.5          1.2         5.8        1.40      105.00       yes           yes
13B/4b         6.5          2.2         7.2        0.75       56.54       yes           yes
34B/4b        17.0          5.7        12.5        0.29       21.62       yes           yes
70B/4b        35.0         11.7        21.5        0.14       10.50       yes           yes
405B/4b      202.5         67.5       105.2        0.02        1.81        no            no
```

Interpretation:

- 7B-13B is comfortable.
- 34B-70B is plausible only with sparse selective loading.
- 405B is not credible on this class of machine.

### Workstation Gen5 NVMe, 128 GB DRAM

Command:

```powershell
python computational_storage_poc\disk_llm_estimator.py --hardware workstation_gen5
```

Observed output:

```text
Disk-resident LLM feasibility (workstation_gen5)
model      size_gb  resident_gb  req_ram_gb  dense_tps<=  flash_tps<=  fits_ram  fits_2x_dram
7B/4b          3.5          1.2         5.8        2.80      210.00       yes           yes
13B/4b         6.5          2.2         7.2        1.51      113.08       yes           yes
34B/4b        17.0          5.7        12.5        0.58       43.24       yes           yes
70B/4b        35.0         11.7        21.5        0.28       21.00       yes           yes
405B/4b      202.5         67.5       105.2        0.05        3.63       yes           yes
```

Interpretation:

- 70B-class looks feasible in principle.
- 405B-class only becomes barely plausible and still looks bandwidth-bound.
- dense streaming remains dead on arrival.

### Dual-NVMe Workstation, 192 GB DRAM

Command:

```powershell
python computational_storage_poc\disk_llm_estimator.py --hardware dual_nvme_workstation
```

Observed output:

```text
Disk-resident LLM feasibility (dual_nvme_workstation)
model      size_gb  resident_gb  req_ram_gb  dense_tps<=  flash_tps<=  fits_ram  fits_2x_dram
7B/4b          3.5          1.2         5.8        4.80      360.00       yes           yes
13B/4b         6.5          2.2         7.2        2.58      193.85       yes           yes
34B/4b        17.0          5.7        12.5        0.99       74.12       yes           yes
70B/4b        35.0         11.7        21.5        0.48       36.00       yes           yes
405B/4b      202.5         67.5       105.2        0.08        6.22       yes           yes
```

Interpretation:

- multi-drive bandwidth scaling helps a lot more than trying to push dense compute into a commodity SSD controller
- even then, 405B remains a low-throughput system

### 3-bit compression scenarios

Command:

```powershell
python computational_storage_poc\disk_llm_estimator.py --hardware workstation_gen5 --bits 3 --models 70 405 1000
```

Observed output:

```text
Disk-resident LLM feasibility (workstation_gen5)
model      size_gb  resident_gb  req_ram_gb  dense_tps<=  flash_tps<=  fits_ram  fits_2x_dram
70B/3b        26.2          8.8        17.1        0.37       28.00       yes           yes
405B/3b      151.9         50.6        79.9        0.06        4.84       yes           yes
1000B/3b     375.0        125.0       191.5        0.03        1.96        no            no
```

Interpretation:

- 3-bit compression meaningfully helps
- trillion-parameter dense-class models still become impractical on a single-node workstation

## What This Means For "Model Size On Disk"

Weight-only storage, ignoring metadata and tokenizer artifacts:

- 7B at 4-bit: about `3.5 GB`
- 13B at 4-bit: about `6.5 GB`
- 34B at 4-bit: about `17 GB`
- 70B at 4-bit: about `35 GB`
- 405B at 4-bit: about `202.5 GB`
- 405B at 3-bit: about `151.9 GB`
- 1T at 3-bit: about `375 GB`

In practice add:

- manifest and bundle metadata
- predictor parameters
- KV cache
- tokenizer / runtime assets
- some padding / alignment overhead

So a safe planning buffer is at least `+10%` on top of weight-only numbers.

## Hardware Impact

### SSD

This design is SSD-bandwidth sensitive first, compute-sensitive second.

What matters most:

- sustained random-to-sequential mixed read throughput
- queue depth support
- read chunk efficiency at `32 KiB+`
- predictable latency under concurrency

### DRAM

You still need real DRAM.
This is not "no-memory inference."

For the paper-like path, DRAM must hold:

- the resident attention slice
- rolling sparse FFN cache / window
- activations
- runtime buffers
- KV cache

### CPU

A disk-first design without a CPU-side low-bit kernel wastes the storage savings.
The CPU path needs:

- direct low-bit execution
- lookup-table or otherwise fused mixed-precision kernels
- async prefetch and overlap between SSD and compute

## Necessary Requirements

### Minimum technical requirements for a serious prototype

1. A transformer microbenchmark, not only MLP/digits proof code.
2. Real disk-backed reads instead of preloading the whole file into RAM.
3. Quantized packing format at `2-4` bits.
4. A sparse predictor plus cache/window policy.
5. CPU kernels that avoid naive dequantize-then-matmul.
6. Token/sec and latency instrumentation under single-stream and concurrent load.

### Minimum hardware requirements by ambition

For `7B-13B`:

- 32 GB RAM
- a good Gen4 NVMe SSD
- modern multi-core CPU

For `34B-70B`:

- 64-128 GB RAM
- Gen5 NVMe preferred
- strong CPU memory bandwidth
- ideally multiple SSDs if concurrency matters

For `405B`:

- roughly 128 GB RAM class if the sparse-flash assumptions hold
- multi-drive bandwidth
- acceptance that throughput is still modest

## Total Output Scaling Ability

The design scales output primarily through **aggregate storage bandwidth** and secondarily through CPU throughput.

Practical consequences:

1. Single-stream token generation is bounded by per-token bytes fetched from disk.
2. Multi-stream concurrency quickly becomes a storage queueing problem.
3. Multi-drive sharding is more valuable than pretending a single commodity SSD controller will become a transformer accelerator.

Rule of thumb from the estimator:

- 70B/4b on a good single Gen5 workstation can land in the low tens of token/s only if the sparse flash assumptions really hold.
- 405B/4b is likely a single-digit token/s system unless storage bandwidth increases materially.

## Recommended Repo Improvements

### Highest-value changes

1. Replace fixed-size dense block packing with a manifest-driven packed format.
   - variable tensor shapes
   - quantized payloads
   - row/column bundles
   - explicit shard placement

2. Add real disk-backed loading.
   - `mmap`
   - async prefetch
   - queue-depth benchmarks
   - no full-file preload in the "NVMe" path

3. Add a sparse transformer path.
   - ReLU or similar sparse FFN retraining
   - small predictor per layer
   - sliding window cache

4. Add CPU low-bit kernels.
   - T-MAC-style LUT execution is the most relevant direction
   - this is a better near-term target than SSD-controller compute

5. Add a real acceptance benchmark.
   - 1B or 3B transformer
   - weight-only `2-4` bit path
   - disk-backed token generation
   - measured tok/s, wattage, and latency

### Cleanup of existing modeling assumptions

1. `mock_nvme.py` should stop assuming theoretical SSD-side MAC throughput without a concrete execution substrate.
2. `mock_array.py` should move from abstract node racing to actual layer/expert sharding simulations.
3. The repo should separate:
   - transport proof
   - storage replay proof
   - disk-resident LLM benchmark

Those are different maturity levels and should stay separately reported.

## Feasibility Verdict

### Feasible

- disk-resident `7B-13B` models on CPU with good SSDs
- disk-resident `34B-70B` if you adopt sparse selective loading plus low-bit CPU kernels
- multi-drive scaling as a bandwidth play

### Conditionally feasible

- `405B` on a large workstation, but only at modest throughput and only with aggressive compression plus sparse loading

### Not currently credible

- dense full-model per-token SSD streaming
- "GPU replacement" via commodity SSD controllers
- trillion-parameter dense-class models on a single workstation without radical compression and bandwidth scaling

## Recommended Next Experiment

Build a `1B-3B` transformer test track inside `computational_storage_poc/` with:

1. 4-bit packed weights on disk
2. one resident-vs-streamed manifest
3. `mmap`-backed layer loading
4. a simple sparse FFN predictor
5. CPU execution only
6. measured tok/s against:
   - baseline CPU full-RAM
   - naive SSD streaming
   - sparse SSD streaming

That will test the theory in a way this repo can honestly defend.
