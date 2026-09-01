# Embedding models on PARAPHRASUS

An extension that benchmarks pre-trained embedding models alongside the existing LLM
and classifier methods. Instead of asking a model to judge each pair, it embeds every
text once and turns the resulting vectors into decisions.

## Why it is a separate pipeline

The LLM benchmark fuses scoring and deciding into one call. Embeddings split cleanly
into two stages:

1. expensive and decision-independent: text to vector
2. cheap and decision-dependent: vectors to a yes/no decision

Stage one runs once into a cache; stage two can then be re-run with different scorers
for nearly nothing. Deduplication makes stage one cheaper still: the benchmark holds
76,058 pairs but only 75,695 distinct texts, because the SNLI, ANLI and XNLI direction
files are the same pairs with the sentences swapped.

## Running

```bash
pip install -r requirements.txt
python3 benchmarking_embed.py configs/full_embed_config.json
python3 extract_results.py configs/full_embed_config.json
```

Results land in `benches/<bench_id>/results.json`, in the same format the LLM
benchmark produces, so both can be compared in one table.

## Configuration

```json
{
  "bench_id": "embed_minilm",
  "embedders": [
    { "name": "M-MiniLM", "model_path": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" }
  ],
  "scorers": [
    { "name": "thr", "method": "threshold" },
    { "name": "lr",  "method": "logistic_regression" }
  ],
  "calibration": { "samples_per_dataset": 500, "seed": 42 },
  "batch_size": 64
}
```

An embedder is either a `model_path` loaded with SentenceTransformers or a `module`
and `function` naming your own. Model-specific encoding arguments are passed through:

```json
{ "name": "mE5", "model_path": "intfloat/multilingual-e5-large",
  "encode_kwargs": { "prompt_name": "query" } }
{ "name": "Jina-v3", "model_path": "jinaai/jina-embeddings-v3", "trust_remote_code": true,
  "encode_kwargs": { "task": "text-matching", "prompt_name": "text-matching" } }
{ "name": "mE5-instr-P1", "model_path": "intfloat/multilingual-e5-large-instruct",
  "prompt": "Instruct: ...\nQuery: " }
```

`prompt` is shorthand for prefixing every text; `encode_kwargs` goes straight to
`.encode`. The E5 and Qwen families expect `prompt_name="query"`, and omitting it
degrades their scores. For your own embedder:

```json
{ "name": "mine", "module": "my_embedders", "function": "encode" }
```

The function takes a list of texts and returns one vector per text, which is exactly
what `SentenceTransformer.encode` already does.

Note that a model used with two different prompts produces two different sets of
vectors, so give each its own `name` to keep their caches apart.

## Scorers

| method | features | needs calibration |
| --- | --- | --- |
| `manual_threshold` | cosine similarity | no, takes a `threshold` |
| `threshold` | cosine similarity | yes |
| `logistic_regression` | unsigned difference `abs(u - v)` | yes |

Both calibrated scorers work on L2-normalized embeddings. The threshold imposes a
monotonic rule on one variable; the logistic regression weights individual embedding
dimensions, so its decision boundary is not monotonic in cosine similarity.

`logistic_regression` also accepts `signed_difference`, `concatenation` and `sum` via
a `feature` key, for comparison against the unsigned difference.

To write your own, subclass `Scorer` in `scorers.py` and point a config entry at a
factory function:

```json
{ "name": "mine", "module": "my_scorers", "function": "build" }
```

## Calibration

Calibrated scorers are fitted leave-one-dataset-out. For each of the ten evaluation
groups, the scorer is fitted on up to `samples_per_dataset` labelled pairs drawn from
each of the other nine, then used to predict that group. No group is ever predicted by
a fit that saw it.

Sampling is uniform over a group's eligible records, without stratification by label,
and happens once per group rather than once per fold, so a group contributes the same
pairs to every fold it appears in. STS-H and TRUE fall below the default cap at 338 and
167 pairs and contribute in full. A typical fold is around 4,000 pairs, roughly 28%
positive.

Fitted parameters, the seed and the calibration size are written to
`benches/<bench_id>/calibration.json`, since an error rate cannot be interpreted
without knowing what its threshold was fitted on.

## Reproducing the embedding paper's numbers

The default profile follows PARAPHRASUS: its group definitions, its score windows, and
its averaging. That makes results directly comparable with the XLM-R and Llama3
baselines in `benches/paper`.

The embedding paper's own pipeline differs in four ways, all of which change the
numbers. `paper_compat` mode reproduces them, and each is toggled separately so you can
isolate which one moves which figure. Copy the config, give it a new `bench_id` so the
two sets of results stay apart, and set the four keys:

```json
{
  "bench_id": "paper_compat",
  "paper_compat": true,
  "scorers": [
    { "name": "thr", "method": "threshold", "objective": "f1" },
    { "name": "lr",  "method": "logistic_regression", "max_iter": 100 }
  ]
}
```

```bash
python3 benchmarking_embed.py configs/paper_compat.json
python3 extract_results.py configs/paper_compat.json
```

The embedding cache is shared, so a second run over models you have already embedded
costs only the scoring stage.

**Group definitions** (`"paper_compat": true`). That pipeline iterates over dataset
*files*, so each NLI direction is its own held-out unit, contributes its own
calibration sample, and counts separately in the Minimize average, which is therefore
over seven files rather than five groups. It also has no STS group: the STS benchmark
is used only for STS-H. And it applies no score window to SICK, labelling all 9,927
pairs non-paraphrase rather than the 2,305 falling in [1, 3) — the excluded set
includes 3,718 pairs scoring 4.0 or above, which a similarity method will call
paraphrases, so this raises the SICK error substantially.

One flag covers all three, because the group definition drives the rotation, the pools
and the averaging together.

**Threshold objective** (`"objective": "f1"`). Their threshold maximizes F1 over a
fixed 200-point grid on [-1, 1], comparing strictly. The default minimizes 0-1 error
exactly over observed midpoints. F1 ignores true negatives, so on a calibration set
that is roughly 70% negative it settles on a lower threshold, predicting more pairs
positive: better on Maximize, worse on Minimize.

**Logistic regression iterations** (`"max_iter": 100`). Their classifier uses
scikit-learn's defaults. On several hundred embedding dimensions, lbfgs often has not
converged by 100 iterations. The default here is 1000.

Results from the two profiles are not comparable with each other, and only the default
is comparable with the published baselines. `calibration.json` records which profile
produced a run.

## Output layout

```
embeddings_cache/                shared across runs, safe to delete
  text_index.json                content hash -> row id
  <embedder>.npy                 memmapped vectors
  <embedder>.mask                which rows are filled
benches/<bench_id>/
  <dataset>.json                 original format, plus "<embedder>/<scorer>": bool
  similarities/<dataset>.json    cosine similarity per record per embedder
  calibration.json               fitted parameters per fold
  results.json                   error rates
```

Similarities are kept out of the prediction files on purpose. Result extraction matches
prediction keys by prefix and compares each value against the expected label, so a
float stored under a matching key would be silently counted as a wrong prediction.

## Pinned versions, and why

`transformers` must stay below v5. Model code published on the Hub was largely written
against v4 and commonly imports symbols that v5 removed, so anything loaded with
`trust_remote_code` fails with an ImportError that does not mention the real cause.
In this benchmark that is mGTE, Jina v3 and KaLM-Emb; every other model loads fine,
which makes the pattern look like three unrelated failures rather than one cause. See
transformers issues #44561 and #45020.

`sentence-transformers` is pinned below 6.0 because 6.x requires transformers v5.
Version 5.7 is the release that added `torch.compile` inference, so nothing is lost.

`tokenizers` has to be pinned explicitly as well. transformers requires
`tokenizers<=0.23.0`, but sentence-transformers asks only for `>=0.19`, so a resolver
is free to leave a newer tokenizers in place. transformers then refuses to import at
all, with a message naming tokenizers rather than anything to do with this benchmark.
Install the three in a single command so they are resolved together.

If you install into an environment that already has these packages loaded, the change
may not take effect until the interpreter restarts.

Before a long run it is worth loading each model once, which catches a broken
environment or a bad model path in minutes rather than after hours of embedding:

```python
import json
from embedders import get_embedder, embedder_check, release_embedder

for entry in json.load(open("configs/full_embed_config.json"))["embedders"]:
    embedder = get_embedder(entry)
    print(entry["name"], embedder_check(entry["name"], embedder), "dims")
    release_embedder(embedder)
```

## When a model fails

A failing embedder does not stop the run. It is logged with a full traceback, recorded
under `failed_embedders` in `calibration.json`, and the benchmark moves on to the next
one. At the end a summary names everything that failed. The run only raises if every
embedder failed, since then there is nothing to extract.

Failures are caught in two places: during up-front validation, which covers a model
that will not load, and during the run itself, which covers things like running out of
memory partway through. Predictions are written out group by group, so work completed
before a failure survives; re-running the same command retries only what is missing.

Models are loaded on first use and released once their embedder finishes, so a
benchmark with many models does not hold them all in memory at once.

## Resuming

Runs resume at two levels. Embeddings are tracked per row in the cache mask, so an
interrupted run recomputes only missing vectors. Predictions are tracked by the
presence of the method key on a record, matching how the LLM benchmark resumes. Re-run
the same command with the same `bench_id`.

The cache is keyed by text content, not by run, so it stays valid across bench ids and
across changes to which datasets you evaluate. Pass `purge_results=True` to drop
predictions while keeping the cache.

## Two things worth knowing

Cosine similarity and the unsigned difference are both symmetric, so an embedding
method returns identical predictions for `pre_hyp` and `hyp_pre`. The benchmark's
directionality probe cannot distinguish anything for these methods. This is a property
of the approach, not a bug, but it makes those two figures non-independent.

The STS benchmark file backs two groups: STS-H, whose 338 records score 4.0 to 5.0, and
STS, whose 706 records fall below 3.0. They are disjoint, so each record is predicted by
exactly one fold, and 335 records belong to neither group and get no prediction.

## Files

| file | role |
| --- | --- |
| `benchmarking_embed.py` | entry point; the two-stage run loop |
| `embedders.py` | resolving and validating embedders |
| `embedding_store.py` | text index and resumable vector cache |
| `similarity.py` | filling the cache; cosine over cached vectors |
| `scorers.py` | features, fit, predict |
| `calibrate_embeddings_clf.py` | leave-one-dataset-out calibration sampling |
| `labels.py` | group definitions and ground-truth labels, shared with `extract_results.py` |
