# Workload Traces

Replay traces for the throughput (RQ1), trace-driven (RQ3), and scalability
(RQ4) experiments. Each line is one request:

```json
{"timestamp": 0.0, "adapter_id": 5, "input_text": "...", "input_tokens": 13, "output_tokens": 112}
```

`timestamp` is the arrival offset in seconds; `adapter_id` selects the LoRA
adapter; `input_text` is the prompt sent to the system under test.

| Directory | Contents | Used by |
|-----------|----------|---------|
| `3min_poisson/` | Poisson arrivals at 1, 2, 4, 8, 12 RPS | `run_throughput.sh` |
| `10min_10adp/` | Five traffic categories, 10 min each | `run_trace_driven.sh` |
| `Scalability/` | Fixed 3 RPS, adapter pools of 10–500 | `run_scalability.sh` |

## Provenance

**Prompts and token lengths** are sampled from the
[ShareGPT V3 dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
(median input 20 tokens, output 299), the conventional workload for LoRA
serving evaluations.

**Arrival timestamps** in `10min_10adp/` are replayed from the Azure Functions
2021 trace (1.98M invocations over 14 days), released with:

> Yanqi Zhang, Íñigo Goiri, Gohar Irfan Chaudhry, Rodrigo Fonseca, Sameh
> Elnikety, Christina Delimitrou, and Ricardo Bianchini. Faster and Cheaper
> Serverless Computing on Harvested Resources. *SOSP '21*.
> https://doi.org/10.1145/3477132.3483580

Ten-minute windows were classified by the coefficient of variation of
inter-arrival times (CV = σ/μ) into steady (CV ≤ 1), normal (1 < CV ≤ 4), and
bursty (CV > 4); combined with a light/heavy load threshold, this yields the
five categories. Arrivals in `3min_poisson/` and `Scalability/` are synthetic
Poisson, not replayed.

## A note on content

`input_text` carries real user prompts as they appear in ShareGPT, which is a
corpus of conversations that users chose to publish. Some retain personal
details their authors included. Nothing was added here, and no attempt was
made to re-identify anyone; the prompts are kept verbatim because prompt
length and content determine prefill cost, so altering them would change the
workload being measured. Anyone reusing these traces inherits ShareGPT's own
terms.
